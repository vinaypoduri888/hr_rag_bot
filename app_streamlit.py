import json
import time
import streamlit as st
from typing import List, Dict, Any
from rag.pipeline import rag_chat, get_store
from rag.config import GEMINI_API_KEY
import warnings
warnings.filterwarnings("ignore", message="Tried to instantiate class '__path__._path'")
# -------------- Page Setup --------------
st.set_page_config(
    page_title="HR RAG Assistant",
    page_icon="🧠",
    layout="wide",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "HR Resource Query Chatbot — FAISS + SentenceTransformers + Gemini Pro Developed by vinay poduri."
    },
)

# -------------- Minimal Styling --------------
CUSTOM_CSS = """
<style>
/* Global tweaks */
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
/* Chat bubbles */
.chat-bubble {
  border-radius: 14px;
  padding: 12px 14px;
  margin: 6px 0 10px 0;
  border: 1px solid rgba(120,120,120,0.15);
  background: rgba(250,250,250,0.75);
}
.chat-bubble.user { background: rgba(100,149,237,0.10); border-color: rgba(100,149,237,0.22); }
.chat-bubble.assistant { background: rgba(60,179,113,0.10); border-color: rgba(60,179,113,0.22); }
.smallcaps { font-variant: all-small-caps; letter-spacing: .5px; color: #666; font-size: 12px;}
.kbd {font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background:#eee; padding:2px 6px; border-radius:6px;}
.badge {display:inline-block; padding:2px 8px; border-radius: 999px; border:1px solid rgba(120,120,120,.2); font-size:12px; margin-right:6px;}
hr.soft { border: none; height: 1px; background: linear-gradient(to right, transparent, rgba(120,120,120,.25), transparent); }
.footer { color:#888; font-size: 12px; text-align:center; margin-top: 14px; }
.tag { display:inline-block; margin-right:6px; margin-bottom:6px; padding:4px 8px; border-radius:999px; border:1px solid rgba(120,120,120,.25); font-size:12px;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------------- Sidebar --------------
with st.sidebar:
    st.markdown("## 🧠 HR RAG Assistant")
    st.caption("Developed by vinay poduri")

    # Status
    st.markdown("### Status")
    try:
        store = get_store()
        idx_ready = store.is_ready()
    except Exception as e:
        idx_ready = False
        st.error(f"Index load failed: {e}")

    st.markdown(
        f"- Vector Index: {'✅ Ready' if idx_ready else '❌ Not found'}\n"
        f"- Gemini Key: {'✅ Set' if GEMINI_API_KEY else '⚠️ Not set (fallback mode)'}"
    )

    # Controls
    st.markdown("### Controls")
    top_k = st.slider("Top-K candidates", min_value=3, max_value=10, value=5, help="How many candidates to consider after hybrid retrieval.")
    show_debug = st.checkbox("Show retrieval debug", value=False)

    st.markdown("### Features")
    st.markdown(
        "- Hybrid retrieval (semantic + filters)\n"
        "- FAISS vector search on CPU\n"
        "- Gemini Pro for HR-grade answers (fallback if no key)\n"
        "- Export results (JSON / Markdown)\n"
        "- Streamlit chat UI + FastAPI backend"
    )

    st.markdown("### How this helps HR")
    st.markdown(
        "- Find best-fit people **fast**\n"
        "- Filter by **skills, experience, availability**\n"
        "- Summarized **justifications** for choices\n"
        "- Repeatable, auditable search"
    )

    st.markdown("---")
    st.markdown("**Quick prompts**")
    qp1, qp2 = st.columns(2)
    if qp1.button("Python · 3+ yrs · Healthcare", use_container_width=True):
        st.session_state._quick_prompt = "Find Python developers with 3+ years for a healthcare project"
    if qp2.button("React Native · Available", use_container_width=True):
        st.session_state._quick_prompt = "Suggest people for a React Native project who are available"

# -------------- Session State --------------
if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, Any]] = []  # [{role, content, results}]
if "_quick_prompt" not in st.session_state:
    st.session_state._quick_prompt = ""

# -------------- Title & Subtitle --------------
st.title(" HR RAG Assistant 🔍")
st.caption("Search people by skills, experience, domain & availability. **Developed by vinay.**")

# -------------- Filters Row --------------
with st.container():
    c1, c2, c3 = st.columns([2,1,1])
    default_query = st.session_state._quick_prompt or "Find Python developers with 3+ years, healthcare"
    query = c1.text_input("Your query", value=default_query, placeholder="e.g., ML + healthcare, 5+ yrs, available")

    skill_filter = c2.text_input("Optional skill filter", value="", placeholder="e.g., Python")
    avail = c3.selectbox("Availability", options=["(any)", "available", "not available"], index=0)

st.markdown("<hr class='soft'/>", unsafe_allow_html=True)

# -------------- Run Search --------------
run_cols = st.columns([1,1,6])
go = run_cols[0].button("🔎 Search", use_container_width=True)
clear_hist = run_cols[1].button("🧹 Clear", use_container_width=True)

if clear_hist:
    st.session_state.history.clear()
    st.session_state._quick_prompt = ""
    st.toast("Cleared conversation.")

if go and query.strip():
    # augment user query with optional quick filters to help retrieval
    augmented = query
    if skill_filter:
        augmented += f" skill:{skill_filter}"
    if avail != "(any)":
        augmented += f" availability:{avail}"

    with st.spinner("Thinking…"):
        t0 = time.time()
        resp = rag_chat(augmented, top_k=top_k)
        latency = time.time() - t0

    # Save to history
    st.session_state.history.append(
        {"role": "user", "content": query.strip()}
    )
    st.session_state.history.append(
        {"role": "assistant", "content": resp.answer, "results": [r.model_dump() for r in resp.results], "debug": resp.debug, "latency": round(latency, 3)}
    )
    # clear quick prompt once used
    st.session_state._quick_prompt = ""

# -------------- Chat Timeline --------------
for turn in st.session_state.history:
    role = turn["role"]
    if role == "user":
        with st.container():
            st.markdown('<div class="chat-bubble user"><span class="smallcaps">you</span><br/>' + turn["content"] + "</div>", unsafe_allow_html=True)
    else:
        with st.container():
            st.markdown('<div class="chat-bubble assistant"><span class="smallcaps">assistant</span><br/>' + turn["content"] + "</div>", unsafe_allow_html=True)

        # Candidate cards
        results = turn.get("results") or []
        if results:
            st.markdown("**Top Candidates**")
            for item in results:
                e = item["employee"]
                score = item.get("score")
                reasons = item.get("reasons", [])
                cols = st.columns([2,1,2,2,1])
                with cols[0]: st.markdown(f"**{e['name']}**")
                with cols[1]: st.write(f"Exp: {e['experience_years']}y")
                with cols[2]: st.write("Skills:", ", ".join(e['skills']))
                with cols[3]: st.write("Projects:", ", ".join(e['projects']))
                with cols[4]: st.write("Avail:", e['availability'])
                if reasons:
                    st.markdown(" ".join([f"<span class='tag'>{r}</span>" for r in reasons]), unsafe_allow_html=True)
                st.caption(f"Score: {score}")
                st.markdown("<hr class='soft'/>", unsafe_allow_html=True)

        # Debug + export
        with st.expander("Debug & Export"):
            latency = turn.get("latency")
            dbg = turn.get("debug")
            if latency is not None:
                st.write(f"Latency: **{latency}s**")
            if show_debug and dbg:
                st.json(dbg)

            # Export buttons
            export_md = f"### HR RAG Answer\n\n{turn['content']}\n\n---\n\n"
            if results:
                export_md += "### Candidates\n"
                for i, item in enumerate(results, 1):
                    e = item["employee"]
                    export_md += f"{i}. **{e['name']}** — {e['experience_years']} yrs; skills: {', '.join(e['skills'])}; projects: {', '.join(e['projects'])}; availability: {e['availability']}\n"
            st.download_button("⬇️ Export as Markdown", export_md, file_name="hr_rag_answer.md")

            export_json = json.dumps({"answer": turn["content"], "results": results, "debug": dbg}, indent=2)
            st.download_button("⬇️ Export as JSON", export_json, file_name="hr_rag_answer.json", mime="application/json")

# -------------- Footer --------------
st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
st.markdown('<div class="footer"> Developed by <b>vinay poduri</b></div>', unsafe_allow_html=True)
