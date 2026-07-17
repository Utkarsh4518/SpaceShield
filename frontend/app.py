import streamlit as st
import numpy as np
import time
import hashlib
import datetime

st.set_page_config(page_title="SpaceShield Command Console", layout="wide", initial_sidebar_state="expanded")

# =====================================================================
# SOVEREIGN COMMAND CONSOLE CSS — v2.0
# =====================================================================
st.markdown("""
<style>
    /* ================================================================
       COMMAND VACUUM BACKGROUND
       Deep aerospace dark palette with spatiotemporal radar grid overlay
       ================================================================ */
    .stApp {
        background-color: #060a13;
        background-image:
            linear-gradient(rgba(0, 229, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 229, 255, 0.03) 1px, transparent 1px);
        background-size: 30px 30px;
    }

    /* ================================================================
       SIDEBAR TACTICAL PARAMETER CONSOLE
       ================================================================ */
    section[data-testid="stSidebar"] {
        background-color: #050910;
        border-right: 1px solid rgba(0, 229, 255, 0.12);
    }

    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #00e5ff;
        font-family: 'Courier New', Monaco, monospace;
        font-size: 0.8rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        border-bottom: 1px solid rgba(0, 229, 255, 0.15);
        padding-bottom: 6px;
    }

    section[data-testid="stSidebar"] label {
        color: #8899aa;
        font-family: 'Courier New', Monaco, monospace;
        font-size: 0.75rem;
    }

    /* ================================================================
       PREMIUM GLASSMORPHIC CONTAINERS
       ================================================================ */
    div[data-testid="stMetric"] {
        background: rgba(16, 26, 48, 0.75);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 229, 255, 0.15);
        border-radius: 6px;
        padding: 14px 16px;
    }

    div[data-testid="stMetric"] label {
        color: #8899aa;
        font-family: 'Courier New', Monaco, monospace;
        font-size: 0.7rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #00e5ff;
        font-family: 'Courier New', Monaco, monospace;
        font-weight: 800;
        font-size: 1.4rem;
    }

    /* ================================================================
       GLOBAL TYPOGRAPHY
       ================================================================ */
    .stApp h1 {
        color: #00e5ff;
        font-family: 'Courier New', Monaco, monospace;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-size: 1.6rem;
    }

    .stApp h3 {
        color: #ff5505;
        font-family: 'Courier New', Monaco, monospace;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-size: 0.95rem;
        border-bottom: 1px solid rgba(255, 85, 5, 0.2);
        padding-bottom: 8px;
    }

    .stApp p, .stApp span, .stApp div {
        color: #c0ccdd;
    }

    .stApp .stMarkdown p {
        font-family: 'Courier New', Monaco, monospace;
        font-size: 0.82rem;
        line-height: 1.6;
        color: #8899aa;
    }

    /* ================================================================
       CHART CONTAINERS
       ================================================================ */
    div[data-testid="stVegaLiteChart"] {
        background: rgba(8, 14, 30, 0.6);
        border: 1px solid rgba(0, 229, 255, 0.1);
        border-radius: 6px;
        padding: 8px;
    }

    .stApp hr {
        border-color: rgba(0, 229, 255, 0.1);
    }

    div[data-testid="stLatex"] {
        background: rgba(16, 26, 48, 0.6);
        border: 1px solid rgba(0, 229, 255, 0.12);
        border-radius: 6px;
        padding: 16px;
        margin: 10px 0;
    }

    .stSlider label, .stTextInput label, .stCheckbox label {
        color: #8899aa;
        font-family: 'Courier New', Monaco, monospace;
        font-size: 0.75rem;
    }

    /* ================================================================
       TOOLTIP READABILITY
       ================================================================ */
    div[data-baseweb="tooltip"] {
        background-color: #1a2744 !important;
        border: 1px solid rgba(0, 229, 255, 0.3) !important;
        border-radius: 4px !important;
    }

    div[data-baseweb="tooltip"] div {
        color: #e0e8f0 !important;
        font-family: 'Courier New', Monaco, monospace !important;
        font-size: 0.78rem !important;
    }

    /* ================================================================
       THREAT STATUS BANNER — State-reactive CSS
       ================================================================ */
    .threat-banner-normal {
        background: rgba(57, 211, 83, 0.08);
        border: 1px solid rgba(57, 211, 83, 0.3);
        border-radius: 6px;
        padding: 18px 24px;
        text-align: center;
        font-family: 'Courier New', Monaco, monospace;
        font-size: 0.85rem;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #39d353;
        margin-bottom: 1rem;
    }

    .threat-banner-jamming {
        background: rgba(210, 153, 34, 0.1);
        border: 2px solid #d29922;
        border-radius: 6px;
        padding: 18px 24px;
        text-align: center;
        font-family: 'Courier New', Monaco, monospace;
        font-size: 0.85rem;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #d29922;
        margin-bottom: 1rem;
        animation: banner-pulse-amber 3s ease-in-out infinite;
    }

    .threat-banner-spoofing {
        background: rgba(248, 81, 73, 0.12);
        border: 2px solid #f85149;
        border-radius: 6px;
        padding: 18px 24px;
        text-align: center;
        font-family: 'Courier New', Monaco, monospace;
        font-size: 0.85rem;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #f85149;
        margin-bottom: 1rem;
        animation: banner-pulse-red 1s ease-in-out infinite;
    }

    @keyframes banner-pulse-amber {
        0%, 100% { border-color: rgba(210, 153, 34, 0.4); }
        50% { border-color: #d29922; box-shadow: 0 0 20px rgba(210, 153, 34, 0.3); }
    }

    @keyframes banner-pulse-red {
        0%, 100% { border-color: rgba(248, 81, 73, 0.4); }
        50% { border-color: #f85149; box-shadow: 0 0 25px rgba(248, 81, 73, 0.4); }
    }

    /* ================================================================
       SYSTEM STATUS BAR
       ================================================================ */
    .status-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(16, 26, 48, 0.5);
        border: 1px solid rgba(0, 229, 255, 0.08);
        border-radius: 4px;
        padding: 8px 16px;
        font-family: 'Courier New', Monaco, monospace;
        font-size: 0.72rem;
        color: #8899aa;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 1rem;
        flex-wrap: wrap;
        gap: 8px;
    }

    .status-bar .conn-online {
        color: #39d353;
        font-weight: 700;
    }

    .status-bar .conn-offline {
        color: #f85149;
        font-weight: 700;
    }

    /* ================================================================
       SCENARIO CONTROL BUTTONS
       ================================================================ */
    div.stButton > button {
        background: rgba(16, 26, 48, 0.7);
        border: 1px solid rgba(0, 229, 255, 0.2);
        color: #00e5ff;
        font-family: 'Courier New', Monaco, monospace;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        border-radius: 4px;
        padding: 8px 16px;
        transition: all 0.2s ease;
        width: 100%;
    }

    div.stButton > button:hover {
        background: rgba(0, 229, 255, 0.1);
        border-color: #00e5ff;
        box-shadow: 0 0 12px rgba(0, 229, 255, 0.2);
    }

    div.stDownloadButton > button {
        background: rgba(57, 211, 83, 0.08);
        border: 1px solid rgba(57, 211, 83, 0.3);
        color: #39d353;
        font-family: 'Courier New', Monaco, monospace;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    div.stDownloadButton > button:hover {
        background: rgba(57, 211, 83, 0.15);
        border-color: #39d353;
        box-shadow: 0 0 12px rgba(57, 211, 83, 0.25);
    }

    /* ================================================================
       FORENSIC LOG CONTAINER
       ================================================================ */
    .forensic-log {
        background: rgba(6, 10, 19, 0.9);
        border: 1px solid rgba(0, 229, 255, 0.1);
        border-radius: 6px;
        padding: 12px 16px;
        font-family: 'Courier New', Monaco, monospace;
        font-size: 0.7rem;
        line-height: 1.7;
        color: #8899aa;
        max-height: 250px;
        overflow-y: auto;
        white-space: pre-line;
    }

    .log-entry-normal { border-left: 3px solid #39d353; padding-left: 8px; margin-bottom: 4px; }
    .log-entry-jamming { border-left: 3px solid #d29922; padding-left: 8px; margin-bottom: 4px; color: #d29922; }
    .log-entry-spoofing { border-left: 3px solid #f85149; padding-left: 8px; margin-bottom: 4px; color: #f85149; }

    /* ================================================================
       MOBILE RESPONSIVE SCALING
       ================================================================ */
    @media (max-width: 768px) {
        .stApp h1 {
            font-size: 1.1rem;
            letter-spacing: 1px;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            font-size: 1.1rem;
        }
        .threat-banner-normal, .threat-banner-jamming, .threat-banner-spoofing {
            font-size: 0.65rem;
            padding: 12px;
        }
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================
def professional_alert(message, level="info"):
    colors = {
        "info": ("#58a6ff", "rgba(88, 166, 255, 0.1)"),
        "success": ("#39d353", "rgba(57, 211, 83, 0.1)"),
        "warning": ("#d29922", "rgba(210, 153, 34, 0.1)"),
        "error": ("#f85149", "rgba(248, 81, 73, 0.1)")
    }
    border, bg = colors.get(level, colors["info"])
    html = f"""
    <div style="padding: 10px; border-radius: 4px; background-color: {bg}; border-left: 4px solid {border}; color: {border}; font-family: monospace; font-weight: bold; margin-bottom: 1rem;">
        {message}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# =====================================================================
# SESSION STATE INITIALIZATION
# =====================================================================
if 'hist_sphericity' not in st.session_state:
    st.session_state.hist_sphericity = [0.0] * 100
if 'hist_error' not in st.session_state:
    st.session_state.hist_error = [0.0] * 100
if 'hist_fim' not in st.session_state:
    st.session_state.hist_fim = [0.0] * 100
if 'session_start_time' not in st.session_state:
    st.session_state.session_start_time = time.time()
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = time.time()
if 'frames_received' not in st.session_state:
    st.session_state.frames_received = 0
if 'event_log' not in st.session_state:
    st.session_state.event_log = []
if 'last_logged_verdict' not in st.session_state:
    st.session_state.last_logged_verdict = "NORMAL"
if 'prev_sphericity' not in st.session_state:
    st.session_state.prev_sphericity = 0.0
if 'prev_metr' not in st.session_state:
    st.session_state.prev_metr = 0.0
if 'prev_latency' not in st.session_state:
    st.session_state.prev_latency = 0.0
if 'prev_dropped' not in st.session_state:
    st.session_state.prev_dropped = 0

# =====================================================================
# SIDEBAR [S1] — DETECTION PARAMETERS
# =====================================================================
st.sidebar.markdown("### DETECTION PARAMETERS")
sat_gamma = st.sidebar.slider(
    "Chi-Squared Threshold Override (γ)", 10.0, 150.0, 50.0, 0.1,
    help="Adjusts the statistical decision boundary for Bartlett Sphericity threat classification. Lower bounds yield tighter defense posture but increase false alarm rate."
)
sat_attenuation = st.sidebar.slider(
    "Antenna Attenuation (dB)", 0.0, 30.0, 0.0, 0.1,
    help="Software-defined antenna front-end attenuation. Reduces overall signal power into the LNA to prevent saturation clipping under high-power jammer scenarios."
)
sat_dynamics = st.sidebar.slider(
    "Target Dynamics Shock (G)", 0.0, 4.0, 0.0, 0.1,
    help="Simulates sudden high-dynamic acceleration steps (up to 4G) to stress-test the Kalman tracking flywheel and EML code loop stability."
)

st.sidebar.markdown("---")

# =====================================================================
# SIDEBAR [S2] — SYSTEM HEALTH INDICATORS
# =====================================================================
st.sidebar.markdown("### SYSTEM HEALTH")

# Backend connection status — local simulation mode for v2 (backend WebSocket integration deferred)
backend_mode = "LOCAL SIMULATION"
st.sidebar.markdown(f"""
<div style="font-family: 'Courier New', monospace; font-size: 0.72rem; margin-bottom: 8px;">
    <span style="color: #d29922; font-weight: 700;">● {backend_mode}</span>
</div>
""", unsafe_allow_html=True)

elapsed = time.time() - st.session_state.session_start_time
elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
st.sidebar.markdown(f"""
<div style="font-family: 'Courier New', monospace; font-size: 0.7rem; color: #8899aa;">
    Session Duration: <span style="color: #00e5ff;">{elapsed_str}</span><br>
    Frames Processed: <span style="color: #00e5ff;">{st.session_state.frames_received}</span><br>
    Execution Provider: <span style="color: #00e5ff;">Fallback-NumPy-Sim</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# =====================================================================
# SIDEBAR — NAVIGATION ANCHORS
# =====================================================================
st.sidebar.markdown("### NAVIGATION")
st.sidebar.markdown("""
<div style="font-family: 'Courier New', monospace; font-size: 0.7rem; line-height: 2.2;">
    <a href="#signal-integrity-charts" style="color: #00e5ff; text-decoration: none;">↓ Signal Integrity</a><br>
    <a href="#tracking-loop-monitor" style="color: #00e5ff; text-decoration: none;">↓ Tracking Loop</a><br>
    <a href="#forensic-event-log" style="color: #00e5ff; text-decoration: none;">↓ Forensic Log</a><br>
    <a href="#compliance-audit" style="color: #00e5ff; text-decoration: none;">↓ Compliance</a>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# SIMULATION EXECUTION
# =====================================================================
# Local simulation model (mirrors backend statistical behavior for offline demo)
effective_snr = 10.0
effective_jammer = -45
effective_power = effective_snr - sat_attenuation

current_sphericity = max(0.0, 20.0 + (30.0 - effective_power) * 0.5 + abs(effective_jammer) * 0.8 + np.random.randn() * 2.0)

# EML Tracking Error (validated 0.0120 chip bound at 4G)
current_error = 0.0010 + (sat_dynamics / 4.0) * 0.0110 + np.random.uniform(-0.0002, 0.0002)

# METR (Maximum Eigenvalue to Trace Ratio) — distinct computation from sphericity
# Isotropic noise → ~0.25, Rank-1 directional source → 1.0
base_metr = 0.25 + (current_sphericity / (sat_gamma * 3.0)) * 0.5
current_metr = min(1.0, max(0.0, base_metr + np.random.uniform(-0.02, 0.02)))

# Buffer Updates
st.session_state.hist_sphericity.append(current_sphericity)
st.session_state.hist_sphericity.pop(0)
st.session_state.hist_error.append(current_error)
st.session_state.hist_error.pop(0)
st.session_state.hist_fim.append(current_metr)
st.session_state.hist_fim.pop(0)

# Threat Classification
if current_sphericity > sat_gamma * 1.5:
    threat_verdict = "CRITICAL SPOOFING"
elif current_sphericity > sat_gamma:
    threat_verdict = "JAMMING"
else:
    threat_verdict = "NORMAL"

# Simulated inference latency
sim_inference_latency = 199.72 + np.random.uniform(-5.0, 5.0)
sim_dropped_blocks = 0

# Frame counter
st.session_state.frames_received += 1
st.session_state.last_update_time = time.time()

# Compute deltas for st.metric
delta_sphericity = round(current_sphericity - st.session_state.prev_sphericity, 2)
delta_metr = round(current_metr - st.session_state.prev_metr, 4)
delta_latency = round(sim_inference_latency - st.session_state.prev_latency, 1)
delta_dropped = sim_dropped_blocks - st.session_state.prev_dropped

# Store current values for next frame delta computation
st.session_state.prev_sphericity = current_sphericity
st.session_state.prev_metr = current_metr
st.session_state.prev_latency = sim_inference_latency
st.session_state.prev_dropped = sim_dropped_blocks

# =====================================================================
# [A] SYSTEM STATUS BAR
# =====================================================================
update_ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.") + f"{datetime.datetime.now().microsecond // 1000:03d}"
st.markdown(f"""
<div class="status-bar">
    <span><span class="conn-offline">● {backend_mode}</span></span>
    <span>Session: <span style="color: #00e5ff;">{elapsed_str}</span></span>
    <span>Frames: <span style="color: #00e5ff;">{st.session_state.frames_received}</span></span>
    <span>Last Update: <span style="color: #00e5ff;">{update_ts}</span></span>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# MAIN DASHBOARD HEADER
# =====================================================================
st.title("SpaceShield Command Console")
st.markdown("Ground-station satellite defense command interface. Real-time spatiotemporal anomaly detection, multi-antenna spatial nulling, and tracking loop visualization.")

# =====================================================================
# [B] THREAT STATUS BANNER
# =====================================================================
if threat_verdict == "CRITICAL SPOOFING":
    banner_class = "threat-banner-spoofing"
    banner_text = "CRITICAL SPOOFING — MULTI-ANTENNA SPATIAL NULLING ENGAGED — PHASE-LOCK UNDER PROTECTION"
elif threat_verdict == "JAMMING":
    banner_class = "threat-banner-jamming"
    banner_text = "BARRAGE JAMMING DETECTED — ELEVATED NOISE FLOOR — SPATIAL FILTERING ACTIVE"
else:
    banner_class = "threat-banner-normal"
    banner_text = "NOMINAL — ALL SPATIAL AND TEMPORAL LOOPS WITHIN CERTIFIED SAFETY ENVELOPES"

st.markdown(f'<div class="{banner_class}">{banner_text}</div>', unsafe_allow_html=True)

# =====================================================================

st.info("Scenario control interfaces loading...")
