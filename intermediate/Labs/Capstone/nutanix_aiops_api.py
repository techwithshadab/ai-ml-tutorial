
import os, json, pathlib, joblib, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List

warnings.filterwarnings("ignore")

app = FastAPI(title="Nutanix AIOps API", version="1.0.0",
              description="Anomaly detection endpoint for Nutanix CVM telemetry")

# ── Load model at startup ─────────────────────────────────────────────────
MODEL_PATH = pathlib.Path("nutanix_anomaly_model.joblib")
if not MODEL_PATH.exists():
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Run Section 3 first.")

_data     = joblib.load(MODEL_PATH)
_scaler   = _data["scaler"]
_model    = _data["model"]
_features = _data["features"]

FEATURE_DEFAULTS = {f: 0.0 for f in _features}


class TelemetryRequest(BaseModel):
    node:                 str
    cpu_pct:              float
    mem_pct:              float
    disk_read_mbps:       float
    disk_write_mbps:      float
    network_mbps:         float
    iops:                 float
    latency_ms:           float
    stargate_ops_per_sec: float
    # Engineered features — computed server-side if omitted
    cpu_pct_roll5_mean:   Optional[float] = None
    cpu_pct_roll5_std:    Optional[float] = 0.0
    latency_ms_roll5_mean: Optional[float] = None
    latency_ms_roll5_std:  Optional[float] = 0.0
    iops_roll5_mean:      Optional[float] = None
    iops_roll5_std:       Optional[float] = 0.0
    mem_pct_roll5_mean:   Optional[float] = None
    mem_pct_roll5_std:    Optional[float] = 0.0


class HealthResponse(BaseModel):
    status:   str
    model:    str
    features: int

class ModelInfoResponse(BaseModel):
    algorithm:     str
    n_estimators:  int
    contamination: float
    n_features:    int
    features:      List[str]
    offset:        float
    model_path:    str

class DetectionResponse(BaseModel):
    node:                  str
    is_anomaly:            bool
    anomaly_score:         float
    confidence:            float
    severity:              str
    timestamp:             str
    feature_contributions: Dict[str, float]


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "model": "IsolationForest", "features": len(_features)}


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    return {
        "algorithm":     "IsolationForest",
        "n_estimators":  int(_model.n_estimators),
        "contamination": float(_model.contamination),
        "n_features":    len(_features),
        "features":      _features,
        "offset":        round(float(_model.offset_), 5),
        "model_path":    str(MODEL_PATH),
    }


@app.post("/detect", response_model=DetectionResponse)
def detect(req: TelemetryRequest):
    row = {
        "cpu_pct":              req.cpu_pct,
        "mem_pct":              req.mem_pct,
        "disk_read_mbps":       req.disk_read_mbps,
        "disk_write_mbps":      req.disk_write_mbps,
        "network_mbps":         req.network_mbps,
        "iops":                 req.iops,
        "latency_ms":           req.latency_ms,
        "stargate_ops_per_sec": req.stargate_ops_per_sec,
        # Engineered — use provided or compute from raw
        "cpu_pct_roll5_mean":    req.cpu_pct_roll5_mean    or req.cpu_pct,
        "cpu_pct_roll5_std":     req.cpu_pct_roll5_std     or 0.0,
        "latency_ms_roll5_mean": req.latency_ms_roll5_mean or req.latency_ms,
        "latency_ms_roll5_std":  req.latency_ms_roll5_std  or 0.0,
        "iops_roll5_mean":       req.iops_roll5_mean       or req.iops,
        "iops_roll5_std":        req.iops_roll5_std        or 0.0,
        "mem_pct_roll5_mean":    req.mem_pct_roll5_mean    or req.mem_pct,
        "mem_pct_roll5_std":     req.mem_pct_roll5_std     or 0.0,
        "cpu_mem_pressure":      req.cpu_pct * req.mem_pct / 100,
        "io_saturation":         (req.disk_read_mbps + req.disk_write_mbps) / 500,
        "latency_iops_ratio":    req.latency_ms / (req.iops / 1000 + 0.001),
        "stargate_health":       req.stargate_ops_per_sec / 2000,
    }

    df_row   = pd.DataFrame([row])
    X        = df_row[_features].fillna(0)
    X_scaled = _scaler.transform(X)
    score    = float(_model.score_samples(X_scaled)[0])
    is_anom  = bool(_model.predict(X_scaled)[0] == -1)
    offset   = _model.offset_
    conf     = float(np.clip(1 - (score - offset) / (abs(offset) + 1e-9), 0, 1))

    severity = ("critical" if conf > 0.85 and is_anom else
                "warning"  if conf > 0.60 and is_anom else
                "info"     if is_anom else "none")

    # Top-3 feature contributions by z-score
    z_row  = X_scaled[0]
    top_ix = sorted(range(len(_features)), key=lambda i: abs(z_row[i]), reverse=True)[:3]
    contributions = {_features[i]: round(float(z_row[i]), 3) for i in top_ix}

    return DetectionResponse(
        node=req.node,
        is_anomaly=is_anom,
        anomaly_score=round(score, 5),
        confidence=round(conf, 4),
        severity=severity,
        timestamp=datetime.utcnow().isoformat() + "Z",
        feature_contributions=contributions,
    )
