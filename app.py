"""
app.py
======
Float Chat – AI-Powered Conversational Oceanographic Data Explorer

Streamlit dashboard with:
  - Overview dashboard (key metrics, charts)
  - Conversational AI interface (ask questions in plain English)
  - Data explorer (filterable table)
  - Interactive T-S, depth-profile, and time-series charts
  - Geographic float map
  - Query-plan inspection
  - Retrieved RAG context inspection
  - Evaluation runner
  - CSV download
"""

from __future__ import annotations

import sys
import os
import pathlib

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path regardless of CWD
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd

# ---------------------------------------------------------------------------
# Page config — MUST be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Float Chat 🌊",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from config import settings                                     # noqa: E402
from data.generate_dataset import generate_dataset             # noqa: E402
from knowledge_base.documents import DOCUMENTS                 # noqa: E402
from rag.embedder import KnowledgeBaseEmbedder                 # noqa: E402
from ai.gemini_client import GeminiClient                      # noqa: E402
from ai.pipeline import AIPipeline                             # noqa: E402
from visualization import charts                               # noqa: E402
from evaluation.evaluator import Evaluator                     # noqa: E402

# ---------------------------------------------------------------------------
# CSS — ocean-inspired theme
# ---------------------------------------------------------------------------
OCEAN_CSS = """
<style>
  /* Background */
  .stApp { background: linear-gradient(135deg, #e0f7fa 0%, #b3e5fc 50%, #e8f5e9 100%); }
  /* Sidebar */
  section[data-testid="stSidebar"] { background: linear-gradient(180deg, #01579b 0%, #0277bd 100%); }
  section[data-testid="stSidebar"] * { color: #e1f5fe !important; }
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stSlider label { color: #b3e5fc !important; }
  /* Cards */
  .metric-card {
    background: white;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border-left: 4px solid #0288d1;
    margin-bottom: 0.5rem;
  }
  /* Chat bubbles */
  .user-bubble {
    background: #0288d1;
    color: white;
    border-radius: 16px 16px 4px 16px;
    padding: 0.6rem 1rem;
    margin: 0.3rem 0;
    max-width: 80%;
    margin-left: auto;
  }
  .assistant-bubble {
    background: white;
    color: #263238;
    border-radius: 16px 16px 16px 4px;
    padding: 0.6rem 1rem;
    margin: 0.3rem 0;
    max-width: 80%;
    border: 1px solid #b3e5fc;
  }
  /* Mode badge */
  .badge-gemini { background:#1976d2; color:white; padding:2px 8px; border-radius:8px; font-size:0.75rem; }
  .badge-rule   { background:#388e3c; color:white; padding:2px 8px; border-radius:8px; font-size:0.75rem; }
  /* Headings */
  h1, h2, h3 { color: #01579b; }
</style>
"""
st.markdown(OCEAN_CSS, unsafe_allow_html=True)


# ===========================================================================
# Session-state helpers
# ===========================================================================

def _init_state():
    defaults = {
        "chat_history": [],
        "df": None,
        "embedder": None,
        "pipeline": None,
        "gemini": None,
        "rag_ready": False,
        "eval_report": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ===========================================================================
# Data loading (cached)
# ===========================================================================

@st.cache_data(show_spinner="📦 Loading dataset …")
def load_dataset() -> pd.DataFrame:
    if settings.DATASET_PATH.exists():
        df = pd.read_csv(settings.DATASET_PATH)
    else:
        df = generate_dataset()
        settings.DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(settings.DATASET_PATH, index=False)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_resource(show_spinner="🧠 Building RAG index (runs once) …")
def build_embedder() -> KnowledgeBaseEmbedder:
    embedder = KnowledgeBaseEmbedder(model_name=settings.RAG_EMBEDDING_MODEL)
    embedder.build_index(DOCUMENTS)
    return embedder


def get_pipeline(embedder) -> AIPipeline:
    if st.session_state["pipeline"] is None:
        gemini = GeminiClient()
        st.session_state["gemini"] = gemini
        st.session_state["pipeline"] = AIPipeline(
            embedder=embedder,
            gemini=gemini,
            top_k=settings.RAG_TOP_K,
        )
    return st.session_state["pipeline"]


# ===========================================================================
# Sidebar
# ===========================================================================

def render_sidebar(df: pd.DataFrame, embedder) -> str:
    with st.sidebar:
        st.markdown("## 🌊 Float Chat")
        st.markdown("*AI-Powered Oceanographic Data Explorer*")
        st.divider()

        page = st.radio(
            "Navigate",
            [
                "🏠 Overview",
                "💬 Chat Interface",
                "🗺️ Float Map",
                "📊 Data Explorer",
                "📈 Charts",
                "🧪 Evaluation",
            ],
            label_visibility="collapsed",
        )

        st.divider()

        # AI mode indicator
        gemini: GeminiClient = st.session_state.get("gemini")
        if gemini and gemini.is_available:
            st.markdown('<span class="badge-gemini">✨ Gemini AI Active</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-rule">⚙️ Rule-Based Mode</span>', unsafe_allow_html=True)
            if gemini and gemini.init_error:
                st.caption(f"ℹ️ {gemini.init_error}")

        # RAG status
        if embedder.is_ready:
            st.markdown(f"**RAG:** ✅ {embedder.document_count} docs indexed")
        else:
            st.markdown("**RAG:** ⏳ Building …")

        st.divider()
        st.markdown(f"**Dataset:** {len(df)} observations")
        st.markdown("⚠️ *Synthetic demo data*")

    return page


# ===========================================================================
# Page renderers
# ===========================================================================

def page_overview(df: pd.DataFrame):
    st.title("🌊 Float Chat – Overview Dashboard")
    st.caption("⚠️ All data shown is **synthetic demonstration data** – not real Argo observations.")

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Observations", len(df))
    c2.metric("Unique Floats", df["float_id"].nunique())
    c3.metric("Regions", df["region"].nunique())
    c4.metric("Avg Temp (°C)", f"{df['temperature'].mean():.2f}")
    c5.metric("Avg Salinity (PSU)", f"{df['salinity'].mean():.2f}")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(charts.regional_stats_bar(df, "temperature"), use_container_width=True)
    with col2:
        st.plotly_chart(charts.qc_flag_pie(df), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(charts.ts_diagram(df), use_container_width=True)
    with col4:
        st.plotly_chart(charts.time_series(df, "temperature"), use_container_width=True)


def page_chat(df: pd.DataFrame, pipeline: AIPipeline):
    st.title("💬 Conversational AI Interface")
    st.caption(
        "Ask questions about the oceanographic dataset in plain English. "
        "The AI extracts your intent, retrieves scientific context (RAG), "
        "and executes a safe Pandas query plan."
    )

    # Suggested questions
    suggestions = [
        "What is the average temperature by region?",
        "Show me observations in the Caribbean Sea",
        "What is the maximum salinity in the Mediterranean Sea?",
        "Compare temperature across all ocean regions",
        "Count observations with good quality data",
        "What are Argo profiling floats?",
    ]
    st.markdown("**💡 Suggested questions:**")
    suggestion_cols = st.columns(3)
    for i, sug in enumerate(suggestions):
        if suggestion_cols[i % 3].button(sug, key=f"sug_{i}", use_container_width=True):
            st.session_state["_pending_question"] = sug

    st.divider()

    # Chat input
    user_input = st.chat_input("Ask anything about the oceanographic data …")
    pending = st.session_state.pop("_pending_question", None)
    question = pending or user_input

    if question:
        with st.spinner("🤔 Processing …"):
            result = pipeline.process(question, df)

        entry = {"question": question, "result": result}
        st.session_state["chat_history"].append(entry)

    # Render chat history
    for entry in reversed(st.session_state["chat_history"]):
        q = entry["question"]
        r = entry["result"]

        with st.chat_message("user"):
            st.write(q)

        with st.chat_message("assistant", avatar="🌊"):
            mode_badge = (
                '<span class="badge-gemini">Gemini</span>'
                if r.mode == "gemini"
                else '<span class="badge-rule">Rule-Based</span>'
            )
            st.markdown(f"Mode: {mode_badge}", unsafe_allow_html=True)

            if r.error:
                st.error(f"❌ {r.error}")
            else:
                qr = r.query_result
                if qr.get("scalar") is not None:
                    st.metric(
                        label=f"{qr['operation'].upper()} of {r.query_plan.get('variable', '')}",
                        value=f"{qr['scalar']:.4f}",
                    )
                if not qr.get("data", pd.DataFrame()).empty:
                    st.dataframe(qr["data"], use_container_width=True)
                    if len(qr["data"]) > 1:
                        st.plotly_chart(
                            charts.query_result_bar(qr["data"], title=q[:60]),
                            use_container_width=True,
                        )
                st.caption(f"Filters applied: {qr.get('filters_applied','none')} | "
                           f"Rows returned: {qr.get('row_count',0)}")

            with st.expander("🔍 Query Plan", expanded=False):
                st.json(r.query_plan)

            if r.rag_docs:
                with st.expander(f"📚 RAG Context ({len(r.rag_docs)} docs retrieved)", expanded=False):
                    for doc in r.rag_docs:
                        st.markdown(f"**{doc['title']}** (score: {doc.get('score', 0):.3f})")
                        st.caption(doc["content"][:300] + "…")
                        st.divider()

    if st.session_state["chat_history"]:
        if st.button("🗑️ Clear conversation"):
            st.session_state["chat_history"] = []
            st.rerun()


def page_map(df: pd.DataFrame):
    st.title("🗺️ Float Position Map")
    st.caption("⚠️ Synthetic data – positions are randomly generated.")

    region_filter = st.multiselect(
        "Filter by region",
        sorted(df["region"].unique()),
        default=sorted(df["region"].unique()),
    )
    filtered = df[df["region"].isin(region_filter)]
    st.plotly_chart(charts.float_map(filtered), use_container_width=True)
    st.info(f"Showing **{len(filtered)}** observations from **{len(region_filter)}** regions.")


def page_data_explorer(df: pd.DataFrame):
    st.title("📊 Data Explorer")
    st.caption("⚠️ Synthetic demonstration data.")

    with st.expander("🔽 Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            regions = st.multiselect("Region", sorted(df["region"].unique()), default=list(df["region"].unique()))
        with col2:
            qc_flags = st.multiselect("QC Flag", sorted(df["qc_flag"].unique()), default=list(df["qc_flag"].unique()))
        with col3:
            depth_range = st.slider("Depth range (m)", 0.0, float(df["depth"].max()), (0.0, float(df["depth"].max())))

    filtered = df[
        (df["region"].isin(regions)) &
        (df["qc_flag"].isin(qc_flags)) &
        (df["depth"] >= depth_range[0]) &
        (df["depth"] <= depth_range[1])
    ]

    st.metric("Matching Observations", len(filtered))
    st.dataframe(filtered, use_container_width=True, height=400)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download filtered data as CSV",
        data=csv,
        file_name="float_chat_filtered.csv",
        mime="text/csv",
    )


def page_charts(df: pd.DataFrame):
    st.title("📈 Interactive Charts")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🌡️ Temp vs Depth", "🧂 Salinity vs Depth", "🔬 T-S Diagram", "📊 Regional Stats"
    ])

    with tab1:
        region = st.selectbox("Filter region (optional)", ["All"] + sorted(df["region"].unique()), key="td_region")
        fig = charts.temperature_depth_profile(df, None if region == "All" else region)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        region2 = st.selectbox("Filter region (optional)", ["All"] + sorted(df["region"].unique()), key="sd_region")
        fig2 = charts.salinity_depth_profile(df, None if region2 == "All" else region2)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.plotly_chart(charts.ts_diagram(df), use_container_width=True)

    with tab4:
        var = st.selectbox("Variable", ["temperature", "salinity", "pressure", "depth"], key="rs_var")
        st.plotly_chart(charts.regional_stats_bar(df, var), use_container_width=True)


def page_evaluation(df: pd.DataFrame, pipeline: AIPipeline):
    st.title("🧪 Evaluation Framework")
    st.markdown(
        "Run all **32 test questions** through the pipeline and see accuracy metrics. "
        "Results reflect **actual measured performance** – no fabricated numbers."
    )

    if st.button("▶️ Run Evaluation (may take 30–60 s)", type="primary"):
        with st.spinner("Running 32 test questions …"):
            evaluator = Evaluator(pipeline, df)
            report = evaluator.run()
        st.session_state["eval_report"] = report

    report = st.session_state.get("eval_report")
    if report is None:
        st.info("Click **Run Evaluation** to start.")
        return

    st.success(f"✅ Evaluation complete! Mode: **{report.mode}**")

    # Summary metrics
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Intent Accuracy", f"{report.intent_accuracy:.0%}")
    m2.metric("Variable Accuracy", f"{report.variable_accuracy:.0%}")
    m3.metric("Region Accuracy", f"{report.region_accuracy:.0%}")
    m4.metric("Filter Accuracy", f"{report.filter_accuracy:.0%}")
    m5.metric("Plan Accuracy", f"{report.plan_accuracy:.0%}")
    m6.metric("RAG Accuracy", f"{report.rag_accuracy:.0%}")

    st.metric("Avg Latency (ms)", f"{report.avg_latency_ms:.1f}")
    st.metric("Total Questions", report.total)

    st.divider()
    st.subheader("Per-Question Results")
    if report.summary_df is not None:
        st.dataframe(report.summary_df, use_container_width=True, height=600)
        csv = report.summary_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Report CSV", csv, "evaluation_report.csv", "text/csv")


# ===========================================================================
# Main
# ===========================================================================

def main():
    df = load_dataset()
    st.session_state["df"] = df

    embedder = build_embedder()
    st.session_state["embedder"] = embedder
    st.session_state["rag_ready"] = embedder.is_ready

    pipeline = get_pipeline(embedder)

    page = render_sidebar(df, embedder)

    if page == "🏠 Overview":
        page_overview(df)
    elif page == "💬 Chat Interface":
        page_chat(df, pipeline)
    elif page == "🗺️ Float Map":
        page_map(df)
    elif page == "📊 Data Explorer":
        page_data_explorer(df)
    elif page == "📈 Charts":
        page_charts(df)
    elif page == "🧪 Evaluation":
        page_evaluation(df, pipeline)


if __name__ == "__main__":
    main()

