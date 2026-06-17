# Run with: streamlit run nutanix_aiops_app.py
# Install:  pip install streamlit google-generativeai scikit-learn pandas numpy matplotlib joblib
# Requires: GEMINI_API_KEY environment variable or sidebar input

import warnings
warnings.filterwarnings("ignore")

import os, json, time, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import streamlit as st
import google.generativeai as genai
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, average_precision_score)
import random

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
AUDIT_PATH = pathlib.Path("aiops_audit.jsonl")

SEVERITY_COLORS = {
    "critical": "#FF4B4B", "warning": "#FFA500",
    "info": "#FFD700",     "none": "#00C897",
    "P1": "#FF4B4B",       "P2": "#FFA500", "P3": "#00C897",
}

BASE_FEATURES = [
    "cpu_pct", "mem_pct", "disk_read_mbps", "disk_write_mbps",
    "network_mbps", "iops", "latency_ms", "stargate_ops_per_sec",
]

FEATURE_COLS = (
    BASE_FEATURES
    + [f"{c}_roll5_mean" for c in ["cpu_pct", "latency_ms", "iops", "mem_pct"]]
    + [f"{c}_roll5_std"  for c in ["cpu_pct", "latency_ms", "iops", "mem_pct"]]
    + ["cpu_mem_pressure", "io_saturation", "latency_iops_ratio", "stargate_health"]
)

PRESETS = {
    "Custom": None,
    "Normal Traffic":          dict(cpu_pct=35.0, mem_pct=55.0, iops=8000.0,  latency_ms=2.5,  disk_read_mbps=120.0, disk_write_mbps=80.0,  network_mbps=500.0,  stargate_ops_per_sec=2000.0),
    "CPU/Memory Pressure":     dict(cpu_pct=94.0, mem_pct=91.0, iops=19500.0, latency_ms=42.0, disk_read_mbps=118.0, disk_write_mbps=79.0,  network_mbps=495.0,  stargate_ops_per_sec=350.0),
    "Disk I/O Saturation":     dict(cpu_pct=58.0, mem_pct=64.0, iops=24000.0, latency_ms=115.0,disk_read_mbps=115.0, disk_write_mbps=482.0, network_mbps=510.0,  stargate_ops_per_sec=1920.0),
    "Network Storm":           dict(cpu_pct=71.0, mem_pct=59.0, iops=7900.0,  latency_ms=29.0, disk_read_mbps=112.0, disk_write_mbps=76.0,  network_mbps=1960.0, stargate_ops_per_sec=1800.0),
    "Memory Exhaustion":       dict(cpu_pct=44.0, mem_pct=98.0, iops=3150.0,  latency_ms=80.0, disk_read_mbps=106.0, disk_write_mbps=81.0,  network_mbps=515.0,  stargate_ops_per_sec=87.0),
}

TEST_EVENTS = [
    {"label": "cpu_memory_pressure",  "node": "node-1", **PRESETS["CPU/Memory Pressure"]},
    {"label": "disk_io_saturation",   "node": "node-3", **PRESETS["Disk I/O Saturation"]},
    {"label": "network_storm",        "node": "node-4", **PRESETS["Network Storm"]},
    {"label": "memory_exhaustion",    "node": "node-2", **PRESETS["Memory Exhaustion"]},
    {"label": "combined_pressure",    "node": "node-1",
     **dict(cpu_pct=88.0, mem_pct=85.0, iops=16000.0, latency_ms=55.0,
            disk_read_mbps=119.0, disk_write_mbps=320.0, network_mbps=1100.0, stargate_ops_per_sec=420.0)},
]

KB_ARTICLES = [
    {"id": "NX-KB-2201", "title": "Stargate WAL Corruption Recovery",
     "keywords": {"stargate", "wal", "crash", "disk", "stargate_health", "io"},
     "content": ("Symptoms: stargate_ops_per_sec < 200, WAL write failure. "
                 "1) ncli disk list | grep -E 'BAD|FAILED'  "
                 "2) ncli disk remove-start disk-id=<id>  "
                 "3) allssh 'genesis stop stargate && sleep 5 && genesis start stargate'  "
                 "4) watch -n10 'ncli cluster info | grep -i redundancy'")},
    {"id": "NX-KB-1845", "title": "Disk I/O Saturation — Erasure Coding Rebuild",
     "keywords": {"disk", "io", "iops", "latency", "rebuild", "disk_write_mbps", "io_saturation"},
     "content": ("Symptoms: disk_write_mbps > 400, iops > 20000, latency_ms > 50ms. "
                 "1) ncli cluster info | grep -i rebuild  "
                 "2) ncli cluster set-rebuild-rate rebuild-rate-bytes-per-second=52428800  "
                 "3) acli vm.migrate vm_name=<vm> host=<less_loaded_host>")},
    {"id": "NX-KB-3102", "title": "Network Storm Isolation on OVS Bridge",
     "keywords": {"network", "storm", "ovs", "network_mbps", "broadcast", "lacp"},
     "content": ("Symptoms: network_mbps > 1500. "
                 "1) allssh 'ovs-appctl fdb/show br0 | sort -k4 -rn | head -20'  "
                 "2) ovs-vsctl del-port br0 <suspect_port>  "
                 "3) allssh 'ifconfig eth0 | grep -E RX.bytes'")},
    {"id": "NX-KB-2756", "title": "CVM Memory Exhaustion — Cassandra OOM Recovery",
     "keywords": {"memory", "mem_pct", "cassandra", "oom", "cvm", "stargate_health"},
     "content": ("Symptoms: mem_pct > 95%, stargate_ops_per_sec < 200. "
                 "1) allssh 'nodetool status'  "
                 "2) allssh 'genesis stop cassandra && sleep 10 && genesis start cassandra'  "
                 "3) allssh 'nodetool cleanup && nodetool compact'")},
    {"id": "NX-KB-4001", "title": "CPU Overcommit — AHV VM Migration",
     "keywords": {"cpu", "overcommit", "cpu_pct", "mem_pct", "cpu_mem_pressure", "vm"},
     "content": ("Symptoms: cpu_pct > 90% sustained. "
                 "1) acli host.get <host> | grep cpu_ready  "
                 "2) acli vm.migrate vm_name=<vm> host=<target>  "
                 "3) acli vm.update vm_name=<vm> cpu_limit_hz=2000000000")},
]

LLM_SYSTEM = (
    "You are a senior Nutanix L2 AIOps engineer and on-call first responder. "
    "You receive structured anomaly detection results. "
    "Rules: identify the PRIMARY root cause from feature_contributions z-scores; "
    "provide exactly 4 ordered remediation steps each with an actual Nutanix CLI command "
    "(ncli, acli, ncc, allssh, genesis); "
    "severity P1=confidence>0.85, P2=confidence>0.60, P3=otherwise; "
    "respond ONLY with valid JSON, no markdown fences."
)

OUTPUT_SCHEMA = {
    "incident_id": "INC-YYYYMMDD-nodeX",
    "severity": "P1|P2|P3",
    "root_cause": "1-2 sentences identifying the trigger",
    "affected_services": ["stargate", "cerebro"],
    "remediation_steps": [{"step": 1, "action": "string", "command": "ncli ..."}],
    "estimated_impact": "string",
    "auto_resolve_probability": 0.0,
    "kb_article_ids": ["NX-XXXX"],
    "confidence": 0.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# Data & Model helpers
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["cpu_pct", "latency_ms", "iops", "mem_pct"]:
        grp = df.groupby("node")[col]
        df[f"{col}_roll5_mean"] = grp.transform(lambda x: x.rolling(5, min_periods=1).mean())
        df[f"{col}_roll5_std"]  = grp.transform(lambda x: x.rolling(5, min_periods=1).std().fillna(0))
    df["cpu_mem_pressure"]   = df["cpu_pct"] * df["mem_pct"] / 100
    df["io_saturation"]      = (df["disk_read_mbps"] + df["disk_write_mbps"]) / 500
    df["latency_iops_ratio"] = df["latency_ms"] / (df["iops"] / 1000 + 0.001)
    df["stargate_health"]    = df["stargate_ops_per_sec"] / 2000
    return df


def make_row(metrics: dict) -> dict:
    """Expand raw telemetry dict into full feature dict (with defaults for engineered cols)."""
    m = metrics
    return {
        **m,
        "cpu_pct_roll5_mean":    m.get("cpu_pct_roll5_mean",    m["cpu_pct"]),
        "cpu_pct_roll5_std":     m.get("cpu_pct_roll5_std",     0.0),
        "latency_ms_roll5_mean": m.get("latency_ms_roll5_mean", m["latency_ms"]),
        "latency_ms_roll5_std":  m.get("latency_ms_roll5_std",  0.0),
        "iops_roll5_mean":       m.get("iops_roll5_mean",       m["iops"]),
        "iops_roll5_std":        m.get("iops_roll5_std",        0.0),
        "mem_pct_roll5_mean":    m.get("mem_pct_roll5_mean",    m["mem_pct"]),
        "mem_pct_roll5_std":     m.get("mem_pct_roll5_std",     0.0),
        "cpu_mem_pressure":      m["cpu_pct"] * m["mem_pct"] / 100,
        "io_saturation":         (m["disk_read_mbps"] + m["disk_write_mbps"]) / 500,
        "latency_iops_ratio":    m["latency_ms"] / (m["iops"] / 1000 + 0.001),
        "stargate_health":       m["stargate_ops_per_sec"] / 2000,
    }


@st.cache_resource(show_spinner="Training anomaly detector on 1,000 normal samples…")
def get_model():
    np.random.seed(42); random.seed(42)
    nodes = ["node-1", "node-2", "node-3", "node-4"]
    SCENARIOS = [
        {"name": "cpu_memory_pressure",  "cpu_pct": 94,  "mem_pct": 91,  "iops": 19500, "latency_ms": 42,  "stargate_ops_per_sec": 350},
        {"name": "disk_io_saturation",   "cpu_pct": 58,  "mem_pct": 65,  "disk_write_mbps": 480, "latency_ms": 115, "iops": 24000},
        {"name": "network_storm",        "cpu_pct": 72,  "mem_pct": 60,  "network_mbps": 1950, "latency_ms": 28},
        {"name": "memory_exhaustion",    "cpu_pct": 45,  "mem_pct": 98,  "stargate_ops_per_sec": 90, "latency_ms": 78, "iops": 3200},
    ]
    base_t = datetime(2026, 5, 29, 6, 0, 0)
    rows = []
    for i in range(1000):
        n = random.choice(nodes)
        rows.append({"timestamp": base_t + timedelta(minutes=i), "node": n,
                     "cpu_pct": float(np.clip(np.random.normal(35, 10), 5, 65)),
                     "mem_pct": float(np.clip(np.random.normal(55, 8), 20, 75)),
                     "disk_read_mbps": float(np.clip(np.random.normal(120, 30), 10, 300)),
                     "disk_write_mbps": float(np.clip(np.random.normal(80, 20), 5, 200)),
                     "network_mbps": float(np.clip(np.random.normal(500, 100), 50, 900)),
                     "iops": float(np.clip(np.random.normal(8000, 1500), 1000, 14000)),
                     "latency_ms": float(np.clip(np.random.normal(2.5, 0.5), 0.3, 5.0)),
                     "stargate_ops_per_sec": float(np.clip(np.random.normal(2000, 400), 800, 4000)),
                     "label": "normal", "scenario": "normal"})
    for i in range(60):
        n = random.choice(nodes); sc = random.choice(SCENARIOS)
        base = rows[i % 100].copy()
        base.update({"timestamp": base_t + timedelta(minutes=random.randint(0, 1000)),
                     "node": n, "label": "anomaly", "scenario": sc["name"]})
        base.update({k: v for k, v in sc.items() if k != "name"})
        rows.append(base)
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df[BASE_FEATURES] = df[BASE_FEATURES].clip(lower=0)
    df_eng = engineer_features(df)
    df_normal = df_eng[df_eng.label == "normal"]
    scaler = StandardScaler()
    model  = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
    X_norm = df_normal[FEATURE_COLS].fillna(0)
    scaler.fit(X_norm); model.fit(scaler.transform(X_norm))
    X_all = scaler.transform(df_eng[FEATURE_COLS].fillna(0))
    df_eng = df_eng.copy()
    df_eng["anomaly_score"]   = model.score_samples(X_all)
    df_eng["is_anomaly_pred"] = model.predict(X_all) == -1
    offset = model.offset_
    df_eng["confidence"] = np.clip(1 - (df_eng["anomaly_score"] - offset) / (abs(offset) + 1e-9), 0, 1)
    return scaler, model, df_eng


def predict_single(scaler, model, metrics: dict) -> dict:
    row = pd.DataFrame([make_row(metrics)])
    X   = row[FEATURE_COLS].fillna(0)
    Xs  = scaler.transform(X)
    score   = float(model.score_samples(Xs)[0])
    is_anom = bool(model.predict(Xs)[0] == -1)
    offset  = model.offset_
    conf    = float(np.clip(1 - (score - offset) / (abs(offset) + 1e-9), 0, 1))
    sev     = ("critical" if conf > 0.85 and is_anom else
               "warning"  if conf > 0.60 and is_anom else
               "info"     if is_anom else "none")
    zrow    = Xs[0]
    top_ix  = sorted(range(len(FEATURE_COLS)), key=lambda i: abs(zrow[i]), reverse=True)[:3]
    contribs = {FEATURE_COLS[i]: round(float(zrow[i]), 3) for i in top_ix}
    return {"is_anomaly": is_anom, "anomaly_score": round(score, 5),
            "confidence": round(conf, 4), "severity": sev,
            "feature_contributions": contribs,
            "timestamp": datetime.utcnow().isoformat() + "Z"}


def retrieve_runbooks(feature_contributions: dict, top_k: int = 2) -> list:
    query_terms = set()
    for feat, zscore in feature_contributions.items():
        if abs(zscore) > 1.0:
            query_terms.update(feat.lower().split("_"))
    scores = {a["id"]: len(query_terms & a["keywords"]) for a in KB_ARTICLES}
    top_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
    by_id   = {a["id"]: a for a in KB_ARTICLES}
    return [by_id[i] for i in top_ids if scores[i] > 0]


def generate_remediation(detection: dict, api_key: str, model_name: str,
                         runbook_context: str = None) -> dict:
    genai.configure(api_key=api_key)
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    node = detection.get("node", "unknown")
    parts = [
        f"Anomaly detection result:\n{json.dumps(detection, indent=2)}",
        f"\nSuggested incident_id: INC-{ts}-{node}",
        f"\nRequired output schema:\n{json.dumps(OUTPUT_SCHEMA, indent=2)}",
    ]
    if runbook_context:
        parts.insert(1, f"\nRelevant KB runbook context:\n{runbook_context}")
    parts.append("\nRespond with ONLY raw JSON conforming to the schema.")
    prompt = "\n".join(parts)
    inst   = genai.GenerativeModel(model_name=model_name, system_instruction=LLM_SYSTEM)
    resp   = inst.generate_content(prompt,
                 generation_config=genai.GenerationConfig(max_output_tokens=700, temperature=0.1))
    raw    = resp.text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"error": "JSON parse failed", "raw_response": raw[:300]}
    result["_tokens"] = {
        "input":  resp.usage_metadata.prompt_token_count,
        "output": resp.usage_metadata.candidates_token_count,
        "cost_usd": (resp.usage_metadata.prompt_token_count * 0.075 +
                     resp.usage_metadata.candidates_token_count * 0.30) / 1_000_000,
    }
    return result


def append_audit(detection: dict, remediation: dict, node: str, pipeline_ms: int) -> dict:
    entry = {
        "timestamp":    datetime.utcnow().isoformat() + "Z",
        "node":         node,
        "is_anomaly":   detection["is_anomaly"],
        "anomaly_score":detection["anomaly_score"],
        "severity":     detection["severity"],
        "confidence":   detection["confidence"],
        "top_features": detection["feature_contributions"],
        "llm_summary": {
            "incident_id":   remediation.get("incident_id", ""),
            "severity":      remediation.get("severity", ""),
            "root_cause":    remediation.get("root_cause", ""),
            "auto_resolve":  remediation.get("auto_resolve_probability", ""),
            "kb_articles":   remediation.get("kb_article_ids", []),
            "steps_count":   len(remediation.get("remediation_steps", [])),
        },
        "pipeline_ms": pipeline_ms,
    }
    with AUDIT_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def load_audit() -> list:
    if not AUDIT_PATH.exists():
        return []
    return [json.loads(ln) for ln in AUDIT_PATH.read_text().strip().splitlines() if ln]


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nutanix AIOps Pipeline",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Configuration")

api_key = st.sidebar.text_input(
    "GEMINI_API_KEY",
    type="password",
    value=os.environ.get("GEMINI_API_KEY", ""),
    help="Get a free key at aistudio.google.com",
)
model_name = st.sidebar.selectbox(
    "Gemini Model", ["gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash"], index=0
)
if api_key:
    st.sidebar.success("API key set ✅")
else:
    st.sidebar.warning("Set GEMINI_API_KEY to enable LLM steps")

st.sidebar.divider()
st.sidebar.subheader("Pipeline Settings")
alert_threshold = st.sidebar.slider("Alert confidence threshold", 0.50, 0.95, 0.60, 0.05)
rag_enabled     = st.sidebar.checkbox("Enable RAG runbook retrieval", value=True)

st.sidebar.divider()
st.sidebar.info(
    
    "Add those MCP tool calls after `append_audit()` in production."
)

# Load model once (cached)
scaler, model, df_scored = get_model()
offset = model.offset_
st.sidebar.success(f"Model ready ✅  |  {len(FEATURE_COLS)} features")

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview", "🔍 Live Detection", "📦 Batch Pipeline", "📋 Audit Log", "📊 Dashboard"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title("🔬 Nutanix AIOps Pipeline")
    st.markdown("**Module 7 Capstone — Real-time Anomaly Detection + LLM Remediation**")

    st.code("""
CVM Telemetry ──► Feature Engineering ──► Isolation Forest (/detect FastAPI)
                                                    │  is_anomaly = true
                                     [RAG] KB Retrieval (FAISS / keyword)
                                                    │
                                     NutanixRemediationEngine (Gemini)
                                                    │
                                     Structured JSON: root_cause, CLI steps
                                                    │
                                     AuditLogger ──► aiops_audit.jsonl
    """, language="text")

    audit_entries = load_audit()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Anomaly Detector", "Isolation Forest")
    c2.metric("Features", len(FEATURE_COLS))
    c3.metric("LLM Engine", model_name.replace("gemini-", "Gemini "))
    c4.metric("Audit Entries", len(audit_entries))

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Pipeline Components")
        st.table(pd.DataFrame([
            {"Module": "M1 — Data Prep",     "Component": "Feature engineering (rolling stats, composites)", "Status": "✅"},
            {"Module": "M4 — NLP",           "Component": "JSON schema design, confidence pattern",          "Status": "✅"},
            {"Module": "M5 — GenAI + RAG",   "Component": "Gemini API, KB runbook retrieval",               "Status": "✅"},
            {"Module": "M6 — Daily Work",    "Component": "Log summariser → Remediation engine",            "Status": "✅"},
            {"Module": "M7 — Capstone",      "Component": "FastAPI + Streamlit + Audit log",                "Status": "✅"},
        ]))

    with col_b:
        st.subheader("Getting Started")
        st.markdown("""
1. **Set your API key** in the sidebar (GEMINI_API_KEY)
2. Go to **🔍 Live Detection** — pick a failure preset and click Analyse
3. Go to **📦 Batch Pipeline** to process all 5 test scenarios
4. Check **📋 Audit Log** to see the JSONL output
5. Visit **📊 Dashboard** for model performance metrics
        """)
        st.info("💡 All anomaly presets are based on real Nutanix failure patterns: disk I/O saturation, network storms, memory exhaustion.")

    st.divider()
    st.subheader("Normal Operating Ranges")
    st.dataframe(pd.DataFrame([
        {"Metric": "CPU %",             "Normal": "20–55%",        "Anomaly Threshold": "> 85%"},
        {"Metric": "Memory %",          "Normal": "40–70%",        "Anomaly Threshold": "> 90%"},
        {"Metric": "IOPS",              "Normal": "5,000–12,000",  "Anomaly Threshold": "> 18,000"},
        {"Metric": "Write latency (ms)","Normal": "0.5–4 ms",      "Anomaly Threshold": "> 20 ms"},
        {"Metric": "Network (Mbps)",    "Normal": "200–800",       "Anomaly Threshold": "> 1,500"},
        {"Metric": "Stargate ops/s",    "Normal": "1,200–3,000",   "Anomaly Threshold": "< 200 (crash)"},
    ]), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Live Detection
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Live CVM Telemetry Analysis")
    st.info("Adjust the sliders or pick a failure preset, then click **Analyse Telemetry**.")

    # Preset loader
    preset_name = st.selectbox("⚡ Load failure scenario preset", list(PRESETS.keys()), index=0)
    preset_vals = PRESETS.get(preset_name) or PRESETS["Normal Traffic"]

    with st.form("telemetry_form"):
        st.subheader("📟 CVM Telemetry Input")
        fc1, fc2 = st.columns(2)
        node = fc1.selectbox("Node", ["node-1", "node-2", "node-3", "node-4"])

        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        cpu_pct   = r1c1.slider("CPU %",         0.0, 100.0, float(preset_vals["cpu_pct"]),         0.5)
        mem_pct   = r1c2.slider("Memory %",      0.0, 100.0, float(preset_vals["mem_pct"]),         0.5)
        iops      = r1c3.slider("IOPS",          0.0, 30000.0, float(preset_vals["iops"]),         100.0)
        latency   = r1c4.slider("Latency (ms)",  0.0, 150.0, float(preset_vals["latency_ms"]),      0.5)

        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        disk_r   = r2c1.slider("Disk Read (MB/s)",  0.0, 600.0,  float(preset_vals["disk_read_mbps"]),       5.0)
        disk_w   = r2c2.slider("Disk Write (MB/s)", 0.0, 600.0,  float(preset_vals["disk_write_mbps"]),      5.0)
        net_mbps = r2c3.slider("Network (Mbps)",    0.0, 2500.0, float(preset_vals["network_mbps"]),        10.0)
        star_ops = r2c4.slider("Stargate ops/s",    0.0, 5000.0, float(preset_vals["stargate_ops_per_sec"]),50.0)

        submitted = st.form_submit_button("🔍 Analyse Telemetry", type="primary", use_container_width=True)

    if submitted:
        metrics = dict(
            cpu_pct=cpu_pct, mem_pct=mem_pct, iops=iops, latency_ms=latency,
            disk_read_mbps=disk_r, disk_write_mbps=disk_w,
            network_mbps=net_mbps, stargate_ops_per_sec=star_ops,
        )
        t0  = time.perf_counter()
        det = predict_single(scaler, model, metrics)
        det["node"] = node
        det_ms = round((time.perf_counter() - t0) * 1000)

        st.divider()
        st.subheader("Detection Result")
        dc1, dc2, dc3, dc4 = st.columns(4)
        sev_icon = {"critical": "🔴", "warning": "🟡", "info": "🟠", "none": "🟢"}.get(det["severity"], "⚪")
        dc1.metric("Is Anomaly",  "⚠️ YES" if det["is_anomaly"] else "✅ NO")
        dc2.metric("Confidence",  f"{det['confidence']:.1%}")
        dc3.metric("Severity",    f"{sev_icon} {det['severity'].upper()}")
        dc4.metric("Detect time", f"{det_ms} ms")

        # Feature contributions bar chart
        if det["feature_contributions"]:
            st.subheader("Top Deviant Features (z-scores)")
            contribs = det["feature_contributions"]
            fig_c, ax_c = plt.subplots(figsize=(8, 2.5))
            keys = list(contribs.keys())
            vals = list(contribs.values())
            colors = ["#FF4B4B" if v > 0 else "#2196F3" for v in vals]
            ax_c.barh(keys, vals, color=colors, alpha=0.85)
            ax_c.axvline(0, color="black", linewidth=0.8)
            ax_c.set_xlabel("z-score (deviation from normal)")
            ax_c.set_title("Feature Contributions", fontweight="bold")
            ax_c.grid(alpha=0.3, axis="x")
            plt.tight_layout()
            st.pyplot(fig_c, use_container_width=False)
            plt.close(fig_c)

        # LLM Remediation
        if det["is_anomaly"]:
            st.divider()
            st.subheader("🤖 LLM Remediation")

            if not api_key:
                st.warning("⚠️ GEMINI_API_KEY not set — set it in the sidebar to get LLM remediation.")
            else:
                runbook_ctx = None
                if rag_enabled:
                    articles = retrieve_runbooks(det["feature_contributions"], top_k=2)
                    if articles:
                        runbook_ctx = "\n\n".join(f"[{a['id']}] {a['title']}:\n{a['content']}" for a in articles)
                        st.caption(f"📚 RAG retrieved: {', '.join(a['id'] for a in articles)}")

                with st.spinner("Calling Gemini…"):
                    t1  = time.perf_counter()
                    rem = generate_remediation(det, api_key, model_name, runbook_ctx)
                    llm_ms = round((time.perf_counter() - t1) * 1000)

                if "error" in rem:
                    st.error(f"LLM error: {rem['error']}")
                else:
                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric("Incident ID",    rem.get("incident_id", "—"))
                    rc2.metric("LLM Severity",   rem.get("severity", "—"))
                    rc3.metric("Auto-resolve",   f"{rem.get('auto_resolve_probability', 0):.0%}")

                    st.markdown(f"**Root Cause:** {rem.get('root_cause', '—')}")

                    steps = rem.get("remediation_steps", [])
                    if steps:
                        st.markdown("**Remediation Steps:**")
                        for s in steps:
                            if isinstance(s, dict):
                                st.markdown(f"**Step {s.get('step', '?')}:** {s.get('action', '')}")
                                st.code(s.get("command", ""), language="bash")

                    kbs = rem.get("kb_article_ids", [])
                    if kbs:
                        st.caption(f"📖 KB References: {', '.join(kbs)}")

                    tokens = rem.get("_tokens", {})
                    st.caption(
                        f"Tokens — input: {tokens.get('input', 0):,}  "
                        f"output: {tokens.get('output', 0):,}  "
                        f"cost: ${tokens.get('cost_usd', 0):.5f}  "
                        f"LLM latency: {llm_ms} ms"
                    )

                    # Audit
                    total_ms = det_ms + llm_ms
                    append_audit(det, rem, node, total_ms)
                    st.success(f"✅ Logged to audit log ({AUDIT_PATH})")
        else:
            st.success("✅ Telemetry is within normal operating range — no action needed.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Batch Pipeline
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("📦 Batch Pipeline — All 5 Test Scenarios")
    st.markdown("Runs all five Nutanix failure scenarios through the full pipeline at once.")

    if not api_key:
        st.warning("⚠️ Set GEMINI_API_KEY in the sidebar to run the batch pipeline.")
    else:
        if st.button("🚀 Run Batch Pipeline", type="primary", use_container_width=True):
            progress = st.progress(0)
            status   = st.empty()
            batch_results = []
            total_cost    = 0.0

            for i, ev in enumerate(TEST_EVENTS):
                status.text(f"Processing {ev['label']} on {ev['node']} ({i+1}/{len(TEST_EVENTS)})…")
                metrics = {k: v for k, v in ev.items() if k not in ("label", "node")}
                t0  = time.perf_counter()
                det = predict_single(scaler, model, metrics)
                det["node"] = ev["node"]

                runbook_ctx = None
                if rag_enabled and det["is_anomaly"]:
                    arts = retrieve_runbooks(det["feature_contributions"], top_k=2)
                    if arts:
                        runbook_ctx = "\n\n".join(f"[{a['id']}] {a['title']}:\n{a['content']}" for a in arts)

                if det["is_anomaly"]:
                    rem = generate_remediation(det, api_key, model_name, runbook_ctx)
                else:
                    rem = {"root_cause": "No anomaly", "severity": "P3",
                           "auto_resolve_probability": 1.0, "kb_article_ids": [],
                           "remediation_steps": [], "_tokens": {"input": 0, "output": 0, "cost_usd": 0}}

                pipeline_ms = round((time.perf_counter() - t0) * 1000)
                total_cost += rem.get("_tokens", {}).get("cost_usd", 0)
                if det["is_anomaly"]:
                    append_audit(det, rem, ev["node"], pipeline_ms)

                batch_results.append({
                    "Event":        ev["label"],
                    "Node":         ev["node"],
                    "Anomaly":      "⚠️ YES" if det["is_anomaly"] else "✅ NO",
                    "Sev (detect)": det["severity"],
                    "Confidence":   f"{det['confidence']:.1%}",
                    "Sev (LLM)":    rem.get("severity", "—"),
                    "Auto-resolve": f"{rem.get('auto_resolve_probability', 0):.0%}",
                    "Time (ms)":    pipeline_ms,
                    "_rem":         rem,
                })
                progress.progress((i + 1) / len(TEST_EVENTS))

            status.empty()
            progress.empty()
            st.success(f"✅ Processed {len(batch_results)} events  |  Total LLM cost: ${total_cost:.5f}")

            display_df = pd.DataFrame([{k: v for k, v in r.items() if k != "_rem"} for r in batch_results])
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("Detailed Remediation Output")
            for r in batch_results:
                rem = r["_rem"]
                title = f"[{rem.get('severity','?')}] {r['Event']} — {str(rem.get('root_cause', 'N/A'))[:70]}"
                with st.expander(title, expanded=False):
                    steps = rem.get("remediation_steps", [])
                    if steps:
                        for s in steps:
                            if isinstance(s, dict):
                                st.markdown(f"**Step {s.get('step','?')}:** {s.get('action','')}")
                                st.code(s.get("command", ""), language="bash")
                    else:
                        st.info("No remediation steps (no anomaly or parse error).")
                    if rem.get("kb_article_ids"):
                        st.caption(f"KB: {', '.join(rem['kb_article_ids'])}")

            st.divider()
            st.metric("Total LLM Cost (batch)", f"${total_cost:.5f}")
            st.caption("Flash pricing: $0.075/M input tokens, $0.30/M output tokens")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Audit Log
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("📋 Audit Log")

    entries = load_audit()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Entries",  len(entries))
    anomaly_entries = [e for e in entries if e.get("is_anomaly")]
    c2.metric("Anomaly Entries", len(anomaly_entries))
    p1_count = sum(1 for e in entries if e.get("llm_summary", {}).get("severity") == "P1")
    c3.metric("P1 Incidents",   p1_count)
    avg_conf = (sum(e.get("confidence", 0) for e in anomaly_entries) / max(len(anomaly_entries), 1))
    c4.metric("Avg Confidence", f"{avg_conf:.1%}")

    st.divider()

    if not entries:
        st.info("No audit entries yet. Run the Live Detection or Batch Pipeline first.")
    else:
        # Filters
        fc1, fc2, fc3 = st.columns(3)
        sev_filter   = fc1.multiselect("Severity", ["critical","warning","info","none"], default=["critical","warning","info"])
        node_filter  = fc2.multiselect("Node",     ["node-1","node-2","node-3","node-4"],
                                       default=["node-1","node-2","node-3","node-4"])
        anom_only    = fc3.checkbox("Anomalies only", value=True)

        filtered = [
            e for e in entries
            if e.get("severity") in sev_filter
            and e.get("node") in node_filter
            and (not anom_only or e.get("is_anomaly"))
        ]

        rows = []
        for e in filtered:
            rows.append({
                "Timestamp":   e["timestamp"][:19].replace("T", " "),
                "Node":        e["node"],
                "Severity":    e["severity"],
                "Confidence":  f"{e.get('confidence', 0):.1%}",
                "Incident ID": e["llm_summary"].get("incident_id", ""),
                "Root Cause":  (e["llm_summary"].get("root_cause") or "")[:65] + "…",
                "Auto-resolve":f"{e['llm_summary'].get('auto_resolve', '') or ''}",
                "Latency ms":  e.get("pipeline_ms", ""),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.divider()
        with st.expander("📄 Raw JSONL"):
            st.code(AUDIT_PATH.read_text() if AUDIT_PATH.exists() else "", language="json")

        col_dl, col_cl = st.columns(2)
        if AUDIT_PATH.exists():
            col_dl.download_button(
                "⬇️ Download Audit Log",
                data=AUDIT_PATH.read_text(),
                file_name="aiops_audit.jsonl",
                mime="application/jsonlines",
                use_container_width=True,
            )
        if col_cl.button("🗑️ Clear Audit Log", use_container_width=True):
            if AUDIT_PATH.exists():
                AUDIT_PATH.unlink()
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Dashboard
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("📊 Model & Pipeline Dashboard")

    y_true  = (df_scored.label == "anomaly").astype(int)
    y_pred  = df_scored.is_anomaly_pred.astype(int)
    y_score = -df_scored.anomaly_score
    roc_auc = roc_auc_score(y_true, y_score)
    avg_prec = average_precision_score(y_true, y_score)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC",          f"{roc_auc:.4f}")
    m2.metric("Avg Precision",    f"{avg_prec:.4f}")
    m3.metric("Training samples", "1,000 (normal only)")
    m4.metric("Decision offset",  round(float(model.offset_), 4))

    st.divider()

    # Row 1: score dist | confusion | per-scenario recall
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4))

    # Score distribution
    normal_scores  = df_scored[df_scored.label == "normal"]["anomaly_score"]
    anomaly_scores = df_scored[df_scored.label == "anomaly"]["anomaly_score"]
    axes1[0].hist(normal_scores,  bins=40, alpha=0.7, color="steelblue", label="Normal")
    axes1[0].hist(anomaly_scores, bins=20, alpha=0.8, color="crimson",   label="Anomaly")
    axes1[0].axvline(model.offset_, color="black", ls="--", lw=1.5, label="Decision boundary")
    axes1[0].set_xlabel("Anomaly Score"); axes1[0].set_title("Score Distribution", fontweight="bold")
    axes1[0].legend(fontsize=8)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    axes1[1].imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        axes1[1].text(j, i, str(v), ha="center", va="center", fontsize=14, fontweight="bold",
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
    axes1[1].set_xticks([0, 1]); axes1[1].set_yticks([0, 1])
    axes1[1].set_xticklabels(["Normal", "Anomaly"]); axes1[1].set_yticklabels(["Normal", "Anomaly"])
    axes1[1].set_xlabel("Predicted"); axes1[1].set_ylabel("Actual")
    axes1[1].set_title("Confusion Matrix", fontweight="bold")

    # Per-scenario recall
    adf = df_scored[df_scored.label == "anomaly"].copy()
    sc_rec = adf.groupby("scenario")["is_anomaly_pred"].mean().sort_values()
    axes1[2].barh(sc_rec.index, sc_rec.values, color="coral", alpha=0.85)
    axes1[2].set_xlabel("Recall"); axes1[2].set_title("Detection Rate by Scenario", fontweight="bold")
    axes1[2].set_xlim(0, 1)
    for i, v in enumerate(sc_rec.values):
        axes1[2].text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    # Row 2: confidence scatter | feature importance | model params
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))

    # Confidence scatter
    normal_df  = df_scored[df_scored.label == "normal"].sample(min(200, len(df_scored[df_scored.label == "normal"])), random_state=42)
    anomaly_df = df_scored[df_scored.label == "anomaly"]
    axes2[0].scatter(normal_df["anomaly_score"],  normal_df["confidence"],  s=5,  alpha=0.4, c="steelblue", label="Normal")
    axes2[0].scatter(anomaly_df["anomaly_score"], anomaly_df["confidence"], s=20, alpha=0.8, c="crimson",   label="Anomaly")
    axes2[0].axhline(0.60, color="orange", ls="--", lw=1, label="Warning threshold")
    axes2[0].axhline(0.85, color="red",    ls="--", lw=1, label="Critical threshold")
    axes2[0].set_xlabel("Anomaly Score"); axes2[0].set_ylabel("Confidence")
    axes2[0].set_title("Score vs Confidence", fontweight="bold"); axes2[0].legend(fontsize=7)

    # Feature importance (mean abs z-score on anomaly samples)
    from sklearn.preprocessing import StandardScaler as _SS
    X_anom  = anomaly_df[FEATURE_COLS].fillna(0)
    X_anom_s = scaler.transform(X_anom)
    feat_imp = dict(zip(FEATURE_COLS, np.abs(X_anom_s).mean(axis=0)))
    top8 = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:8]
    fn, fv = zip(*top8)
    axes2[1].barh(list(fn), list(fv), color="coral", alpha=0.85)
    axes2[1].set_xlabel("Mean |z-score| on anomaly samples")
    axes2[1].set_title("Top Feature Importance", fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    # Model params table
    st.divider()
    st.subheader("Model Parameters")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.table(pd.DataFrame([
            {"Parameter": "Algorithm",       "Value": "Isolation Forest"},
            {"Parameter": "n_estimators",    "Value": 200},
            {"Parameter": "contamination",   "Value": "0.05 (5%)"},
            {"Parameter": "n_features",      "Value": len(FEATURE_COLS)},
            {"Parameter": "Training size",   "Value": "1,000 normal samples"},
            {"Parameter": "Decision offset", "Value": round(float(model.offset_), 5)},
        ]))
    with col_p2:
        rep = classification_report(y_true, y_pred, target_names=["normal", "anomaly"],
                                    output_dict=True)
        rep_df = pd.DataFrame({
            "Class":    ["Normal", "Anomaly"],
            "Precision":[f"{rep['normal']['precision']:.3f}", f"{rep['anomaly']['precision']:.3f}"],
            "Recall":   [f"{rep['normal']['recall']:.3f}",    f"{rep['anomaly']['recall']:.3f}"],
            "F1":       [f"{rep['normal']['f1-score']:.3f}",  f"{rep['anomaly']['f1-score']:.3f}"],
        })
        st.table(rep_df)
        st.caption(
            "Training is **unsupervised** (normal samples only). "
            "Labels are used for evaluation only — in production there are no anomaly labels at training time."
        )

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "*Module 7 Capstone | AI/ML Intermediate Workshop | Nutanix AIOps Pipeline*  "
    "| Powered by Isolation Forest + Gemini 1.5"
)
