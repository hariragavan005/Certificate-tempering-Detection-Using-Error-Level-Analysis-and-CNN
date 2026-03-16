import streamlit as st
from ultralytics import YOLO
import cv2
import pandas as pd
import tempfile
from PIL import Image

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CertVerify The Certificate Forgery Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# GLOBAL THEME
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:ital,wght@0,200..900;1,200..900&display=swap');
/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
}

.main {
    background: #080c14;
}

.block-container {
    padding: 2rem 2.5rem 4rem;
    max-width: 1100px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #050810;
    border-right: 1px solid #1a2235;
}

section[data-testid="stSidebar"] .stRadio label {
    font-family: 'Source Sans 3', sans-serif !important;
    color: #7a92b8 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em;
    padding: 6px 0;
    transition: color 0.2s;
}

section[data-testid="stSidebar"] .stRadio label:hover {
    color: #38bdf8 !important;
}

/* ── Typography ── */
h1 {
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2.4rem !important;
    color: #f0f6ff !important;
    letter-spacing: -0.02em;
    line-height: 1.15 !important;
}

h2, h3 {
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 700 !important;
    color: #cbd5e1 !important;
    letter-spacing: -0.01em;
}

p, li, .stMarkdown {
    color: #94a3b8 !important;
    font-size: 0.88rem;
    line-height: 1.75;
}

/* ── Metric Cards ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0f172a, #0d1829);
    border: 1px solid #1e3050;
    border-radius: 10px;
    padding: 1.2rem 1.4rem !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.5);
    transition: border-color 0.25s;
}

[data-testid="stMetric"]:hover {
    border-color: #38bdf8;
}

[data-testid="stMetricLabel"] {
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 0.7rem !important;
    color: #475569 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

[data-testid="stMetricValue"] {
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 1.9rem !important;
    font-weight: 800 !important;
    color: #f0f6ff !important;
}

/* ── File uploader ── */
div[data-testid="stFileUploader"] {
    border: 1.5px dashed #1e3a5f;
    border-radius: 12px;
    padding: 1.5rem;
    background: rgba(14, 28, 54, 0.4);
    transition: border-color 0.25s;
}

div[data-testid="stFileUploader"]:hover {
    border-color: #38bdf8;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1e3050 !important;
    border-radius: 10px;
    overflow: hidden;
}

/* ── Divider ── */
hr {
    border-color: #1e3050 !important;
    margin: 1.5rem 0 !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 10px;
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 0.82rem;
}

/* ── Sidebar logo area ── */
.sidebar-brand {
    padding: 0.5rem 0 1.5rem;
    border-bottom: 1px solid #1a2235;
    margin-bottom: 1.5rem;
}

.sidebar-brand h2 {
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 1.15rem !important;
    color: #38bdf8 !important;
    letter-spacing: 0.05em;
    margin: 0 !important;
}

.sidebar-brand p {
    font-size: 0.68rem !important;
    color: #334155 !important;
    margin: 2px 0 0 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ── Page tag ── */
.page-tag {
    display: inline-block;
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 0.65rem;
    color: #38bdf8;
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.25);
    border-radius: 4px;
    padding: 2px 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

/* ── Result verdict boxes ── */
.verdict-fake {
    background: linear-gradient(135deg, #3b0a0a, #1f0505);
    border: 1px solid #7f1d1d;
    padding: 1.2rem 1.6rem;
    border-radius: 12px;
    color: #fca5a5;
    font-family: 'Source Sans 3', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: 0.02em;
}

.verdict-real {
    background: linear-gradient(135deg, #052e1c, #031a10);
    border: 1px solid #065f46;
    padding: 1.2rem 1.6rem;
    border-radius: 12px;
    color: #6ee7b7;
    font-family: 'Source Sans 3', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: 0.02em;
}

/* ── Info card ── */
.info-card {
    background: linear-gradient(135deg, #0f172a, #0d1829);
    border: 1px solid #1e3050;
    border-left: 3px solid #38bdf8;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}

.info-card h4 {
    font-family: 'Source Sans 3', sans-serif;
    color: #e2e8f0 !important;
    font-size: 0.9rem;
    margin: 0 0 4px !important;
}

.info-card p {
    font-size: 0.78rem !important;
    color: #64748b !important;
    margin: 0 !important;
}

/* ── Step badges ── */
.step-row {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    margin-bottom: 1rem;
}

.step-num {
    width: 28px;
    height: 28px;
    min-width: 28px;
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.35);
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Source Sans 3', sans-serif;
    font-weight: 700;
    font-size: 0.8rem;
    color: #38bdf8;
}

.step-text {
    font-size: 0.83rem;
    color: #94a3b8;
    padding-top: 4px;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return YOLO("best (3).pt")

model = load_model()

# ============================================================
# SESSION HISTORY
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def analyze_predictions(results):
    fake_count = 0
    total = 0
    max_conf = 0
    most_suspicious = None
    region_data = []

    for i, box in enumerate(results.boxes):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls]
        total += 1

        if conf > max_conf:
            max_conf = conf
            most_suspicious = box

        if class_name == "fake":
            fake_count += 1

        region_data.append({
            "Region": i + 1,
            "Type": class_name.upper(),
            "Confidence": f"{round(conf * 100, 2)}%"
        })

    return total, fake_count, max_conf, most_suspicious, region_data


def draw_boxes(image_path, results, most_suspicious):
    image = cv2.imread(image_path)

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls]

        if box is most_suspicious:
            color = (255, 140, 0)   # amber = most suspicious
        elif class_name == "true":
            color = (34, 197, 94)   # green = genuine
        else:
            color = (239, 68, 68)   # red = fake

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name.upper()} {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    return image


def calculate_risk(fake_count, total):
    if total == 0:
        return 0, "Unknown"
    risk_score = (fake_count / total) * 100
    if risk_score < 20:
        level = "LOW"
    elif risk_score < 50:
        level = "MEDIUM"
    else:
        level = "HIGH"
    return round(risk_score, 2), level

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h2>🔍 CertVerify</h2>
        <p>AI Forgery Detection</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Home", "Upload & Detect", "History Log"],
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Source Sans 3',sans-serif;font-size:0.65rem;color:#1e3050;
                border-top:1px solid #1a2235;padding-top:1rem;line-height:2;">
        MODEL &nbsp; YOLOv8<br>
        CONF &nbsp;&nbsp; 0.40 threshold<br>
        BUILD &nbsp; v2.0
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# HOME PAGE
# ============================================================

if page == "Home":

    st.markdown('<div class="page-tag">Overview</div>', unsafe_allow_html=True)
    st.title("Certificate Forgery\nDetection System")
    st.markdown("---")

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown("### What this system does")
        st.markdown("""
        This platform uses a trained **YOLOv8** deep learning model to scan
        certificate images and flag potentially forged or tampered regions with
        bounding box visualizations and confidence scores.
        """)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### How to use")

        for step, desc in [
            ("01", "Navigate to **Upload & Detect** from the sidebar."),
            ("02", "Upload a certificate image (JPG, JPEG, or PNG)."),
            ("03", "The model will scan and annotate suspicious regions."),
            ("04", "Review the risk score, region table, and final verdict."),
            ("05", "All results are saved in the **History Log** for the session."),
        ]:
            st.markdown(f"""
            <div class="step-row">
                <div class="step-num">{step}</div>
                <div class="step-text">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("### Capabilities")

        for icon, title, desc in [
            ("🎯", "Region Detection", "Identifies forged bounding boxes with pixel precision"),
            ("📊", "Risk Scoring", "Calculates a 0–100% forgery risk index"),
            ("🖼️", "Visual Overlay", "Draws annotated output over the original image"),
            ("📋", "Region Table", "Lists every detected region with type and confidence"),
            ("🗂️", "Session History", "Stores all detections for the current session"),
        ]:
            st.markdown(f"""
            <div class="info-card">
                <h4>{icon} &nbsp; {title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# UPLOAD & DETECT PAGE
# ============================================================

elif page == "Upload & Detect":

    st.markdown('<div class="page-tag">Detection</div>', unsafe_allow_html=True)
    st.title("Upload & Detect")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Drop a certificate image here or click to browse",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            temp_path = tmp.name

        results = model.predict(temp_path, conf=0.4)[0]

        col_orig, col_det = st.columns(2, gap="medium")

        with col_orig:
            st.markdown("**Original Image**")
            st.image(image, use_container_width=True)

        if len(results.boxes) == 0:
            with col_det:
                st.markdown("**Detection Output**")
                st.image(image, use_container_width=True)

            st.markdown("---")
            st.warning("⚠️ No forgery-related regions detected. Unable to determine authenticity.")

        else:
            total, fake_count, max_conf, most_suspicious, region_data = analyze_predictions(results)
            output_img = draw_boxes(temp_path, results, most_suspicious)
            risk_score, risk_level = calculate_risk(fake_count, total)

            with col_det:
                st.markdown("**Detection Output**")
                st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB),
                         use_container_width=True)

            # ── Metrics ──
            st.markdown("---")
            st.markdown("### Detection Summary")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Detected Regions", total)
            m2.metric("Fake Regions", fake_count)
            m3.metric("Max Confidence", f"{max_conf * 100:.1f}%")
            m4.metric(f"Risk Level", risk_level)

            # ── Risk bar ──
            st.markdown("### Forgery Risk Score")
            risk_color = "#ef4444" if risk_score >= 50 else "#f59e0b" if risk_score >= 20 else "#22c55e"
            st.markdown(f"""
            <div style="background:#0f172a;border:1px solid #1e3050;border-radius:10px;padding:1rem 1.4rem;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                    <span style="font-family:'Source Sans 3',sans-serif;font-size:0.7rem;
                                 color:#475569;text-transform:uppercase;letter-spacing:0.1em;">
                        Risk Index
                    </span>
                    <span style="font-family:'Source Sans 3',sans-serif;font-weight:800;font-size:1.4rem;color:{risk_color};">
                        {risk_score}%
                    </span>
                </div>
                <div style="background:#1e3050;border-radius:6px;height:8px;overflow:hidden;">
                    <div style="width:{risk_score}%;background:{risk_color};
                                height:100%;border-radius:6px;
                                transition:width 0.5s ease;">
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Region table ──
            st.markdown("### Detected Region Details")
            df = pd.DataFrame(region_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # ── Save to history ──
            st.session_state.history.append({
                "File": uploaded_file.name,
                "Total Regions": total,
                "Fake Regions": fake_count,
                "Max Confidence": f"{max_conf * 100:.1f}%",
                "Risk Score": f"{risk_score}%",
                "Risk Level": risk_level
            })

            # ── Final verdict ──
            st.markdown("---")
            if fake_count > 0:
                st.markdown(f"""
                <div class="verdict-fake">
                    🚨 &nbsp; VERDICT: CERTIFICATE APPEARS FAKE
                    <div style="font-family:'Source Sans 3',sans-serif;font-weight:400;
                                font-size:0.75rem;margin-top:6px;color:#fca5a5;opacity:0.7;">
                        {fake_count} suspicious region(s) detected — integrity compromised
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="verdict-real">
                    ✅ &nbsp; VERDICT: CERTIFICATE APPEARS AUTHENTIC
                    <div style="font-family:'Source Sans 3',sans-serif;font-weight:400;
                                font-size:0.75rem;margin-top:6px;color:#6ee7b7;opacity:0.7;">
                        No fake regions detected — all regions classified as genuine
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ============================================================
# HISTORY LOG PAGE
# ============================================================

elif page == "History Log":

    st.markdown('<div class="page-tag">History</div>', unsafe_allow_html=True)
    st.title("Session History")
    st.markdown("---")

    if len(st.session_state.history) == 0:
        st.info("📂 No detections yet in this session. Upload a certificate to get started.")
    else:
        total_scans   = len(st.session_state.history)
        fake_detected = sum(1 for r in st.session_state.history if int(r["Fake Regions"]) > 0)
        genuine       = total_scans - fake_detected

        h1, h2, h3 = st.columns(3)
        h1.metric("Total Scans", total_scans)
        h2.metric("Flagged as Fake", fake_detected)
        h3.metric("Appeared Genuine", genuine)

        st.markdown("### All Scans")
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("""
        <div style="font-family:'Source Sans 3',sans-serif;font-size:0.68rem;
                    color:#334155;margin-top:1rem;">
            ⚠ &nbsp; History is session-scoped and clears when the server restarts.
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑 Clear History"):
            st.session_state.history = []
            st.rerun()