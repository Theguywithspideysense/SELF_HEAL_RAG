import html
from pathlib import Path

import streamlit as st
from langchain_core.documents import Document

from src.graph import graph
from src.loader import extract_documents
from src.vectorstore import split_documents, add_documents, count_documents


st.set_page_config(page_title="RAG", layout="wide", initial_sidebar_state="collapsed")

# patch markdown to strip blank lines from HTML blocks (prevents code-block misparse)
_md = st.markdown
def _safe_md(body, *args, **kwargs):
    if kwargs.get("unsafe_allow_html") and isinstance(body, str):
        body = "\n".join(l for l in body.strip().splitlines() if l.strip())
    return _md(body, *args, **kwargs)
st.markdown = _safe_md


# ---- css ----

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background: #e1d4c2 !important;
    color: #291c0e !important;
    font-family: 'Inter', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] { display: none !important; }

.block-container {
    max-width: 1180px !important;
    margin: 18px auto !important;
    padding: 38px 42px 28px 42px !important;
    background: #e1d4c2;
    border: 1.5px solid #291c0e;
    border-radius: 24px;
    box-shadow: 10px 10px 25px rgba(41,28,14,.18),
                -5px -5px 14px rgba(255,255,255,.55);
}

h1, h2, h3 { color: #291c0e !important; }

.label {
    font-size: 11px; font-weight: 700; letter-spacing: 1.8px;
    text-transform: uppercase; color: #6e4937; margin-bottom: 8px;
}

.page-title {
    font-size: 34px; font-weight: 700; color: #291c0e;
    letter-spacing: -1px; line-height: 1.15; margin: 0 0 8px 0;
}

.page-sub {
    font-size: 14px; color: #3f3025; line-height: 1.6; margin-bottom: 0;
}

.rule {
    height: 1px; background: rgba(41,28,14,.25); margin: 26px 0;
}

.paper, .metric, .answer-block {
    background: #e1d4c2;
    border: 1.5px solid #291c0e;
    box-shadow: 6px 6px 14px rgba(41,28,14,.14),
                -4px -4px 10px rgba(255,255,255,.45);
}

.paper {
    border-radius: 18px; padding: 22px; margin-bottom: 16px;
}
.paper-title { font-size: 16px; font-weight: 700; color: #291c0e; margin-bottom: 6px; }
.paper-sub { font-size: 13px; color: #513d2e; line-height: 1.65; }

.metric {
    border-radius: 17px; padding: 17px 19px; min-height: 105px;
}
.metric-label {
    font-size: 10px; font-weight: 700; letter-spacing: 1.3px;
    text-transform: uppercase; color: #654a39; margin-bottom: 9px;
}
.metric-value { font-size: 19px; font-weight: 700; color: #291c0e; }
.metric-value.hi { color: #6e4937; }
.metric-value.lo { color: #8b7767; }

.answer-block {
    border-radius: 17px; padding: 21px 23px;
    font-size: 14px; line-height: 1.85; color: #291c0e;
}

.chunk {
    background: #e1d4c2; border: 1px solid #291c0e; border-radius: 12px;
    padding: 13px 15px; margin: 8px 0;
    box-shadow: 3px 3px 8px rgba(41,28,14,.12),
                -2px -2px 6px rgba(255,255,255,.35);
}
.chunk-meta { display:flex; justify-content:space-between; margin-bottom:7px; }
.chunk-rank { font-size:10px; font-weight:700; color:#6e4937; }
.chunk-score { font-size:10px; color:#806a5a; }
.chunk-source { font-size:11px; font-weight:700; color:#493528; margin-bottom:5px; }
.chunk-preview { font-size:12px; color:#604b3b; line-height:1.6; }

.qbox {
    background:#ebe1d3; border:1px solid #291c0e; border-radius:11px;
    padding:11px 14px; font-size:12px; color:#493528; line-height:1.6;
    word-break:break-word; box-shadow:inset 2px 2px 5px rgba(41,28,14,.08);
}

.trace-item { border-left:2px solid #8b7767; padding:8px 0 8px 14px; margin:10px 0; }
.trace-item.active { border-left-color:#6e4937; }
.trace-title { font-size:12px; font-weight:700; color:#493528; }
.trace-sub { font-size:11px; color:#806a5a; margin-top:3px; }

.pill-ok, .pill-no {
    display:inline-block; font-size:10px; font-weight:700;
    border-radius:7px; padding:2px 8px; border:1px solid #291c0e;
}
.pill-ok { background:#a78d78; color:#fff; }
.pill-no { background:#d2c3b2; color:#604b3b; }

.pipeline {
    display:flex; flex-wrap:wrap; align-items:center; gap:7px; margin-top:8px;
}
.pnode {
    background:#e1d4c2; border:1px solid #291c0e; border-radius:9px;
    padding:6px 11px; font-size:10px; font-weight:600; color:#604b3b;
}
.pnode.on { background:#a78d78; color:#fff; border-color:#291c0e; }
.parrow { color:#6e4937; font-size:12px; }

.stButton > button {
    background:#a78d78 !important; color:#fff !important;
    border:1.5px solid #291c0e !important; border-radius:12px !important;
    min-height:43px; font-size:13px; font-weight:700;
    box-shadow:4px 4px 9px rgba(41,28,14,.18),
               -3px -3px 7px rgba(255,255,255,.35);
}


[data-testid="stFileUploader"] {
    background:#e1d4c2 !important; border:1.5px dashed #291c0e !important;
    border-radius:14px !important; padding:8px !important;
}

[data-testid="stFileUploader"] button {
    background: #a78d78 !important;
    color: #ffffff !important;
    border: 1.5px solid #291c0e !important;
    border-radius: 10px !important;
}

[data-testid="stFileUploader"] button:hover {
    background: #8f705d !important;
    color: #ffffff !important;
}

textarea {
    background: #ebe1d3 !important;
    color: #6e4937 !important;
    border: 1.5px solid #291c0e !important;
    border-radius: 14px !important;
    font-size: 14px !important;
}

textarea::placeholder {
    color: #6e4937 !important;
    opacity: 0.8 !important;
}

textarea:focus { border-color:#6e4937 !important; }

[data-testid="stAlert"], [data-testid="stExpander"] {
    background:#e1d4c2 !important; color:#291c0e !important;
    border:1px solid #291c0e !important; border-radius:12px !important;
}

@media (max-width:900px) {
    .block-container {
        margin:8px !important; padding:25px 20px !important; border-radius:18px;
    }
    .page-title { font-size:28px; }
}
</style>
""", unsafe_allow_html=True)


# ---- helpers ----

def esc(v):
    return html.escape(str(v))

def rule():
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

def label(text):
    st.markdown(f'<div class="label">{esc(text)}</div>', unsafe_allow_html=True)

def metric(lbl, val, active=False):
    cls = "hi" if active else ""
    return f"""
<div class="metric">
<div class="metric-label">{esc(lbl)}</div>
<div class="metric-value {cls}">{esc(val)}</div>
</div>"""

def pill(ok):
    if ok:
        return '<span class="pill-ok">yes</span>'
    return '<span class="pill-no">no</span>'

def render_chunk(item):
    st.markdown(f"""
<div class="chunk">
<div class="chunk-meta">
<span class="chunk-rank">#{esc(item.get('rank','-'))}</span>
<span class="chunk-score">sim {esc(item.get('score', 0))}</span>
</div>
<div class="chunk-source">{esc(item.get('source','Unknown'))}</div>
<div class="chunk-preview">{esc(item.get('preview',''))}</div>
</div>""", unsafe_allow_html=True)

def render_trace(trace):
    if not trace:
        return
    label("Execution trace")
    for step in trace:
        kind = step.get("step", "")
        retry = step.get("retry", 0)

        if kind == "retrieve":
            n = len(step.get("results", []))
            st.markdown(f"""
<div class="trace-item active">
<div class="trace-title">Retrieval — attempt {retry + 1}</div>
<div class="trace-sub">Query: {esc(step.get('query',''))} &nbsp;|&nbsp; {n} chunks</div>
</div>""", unsafe_allow_html=True)
            with st.expander(f"Chunks — attempt {retry + 1}", expanded=(retry == 0)):
                results = step.get("results", [])
                if not results:
                    st.caption("Nothing retrieved.")
                else:
                    for item in results:
                        render_chunk(item)

        elif kind == "generate":
            st.markdown(f"""
<div class="trace-item">
<div class="trace-title">Generation — attempt {retry + 1}</div>
<div class="trace-sub">qwen2.5-coder:7b produced an answer from context.</div>
</div>""", unsafe_allow_html=True)

        elif kind == "critic":
            grounded = step.get("grounded", False)
            sufficient = step.get("sufficient_context", False)
            st.markdown(f"""
<div class="trace-item">
<div class="trace-title">Critic — attempt {retry + 1}</div>
<div class="trace-sub">
Grounded: {pill(grounded)} &nbsp; Context: {pill(sufficient)}
</div>
</div>""", unsafe_allow_html=True)
            with st.expander(f"Critic analysis — attempt {retry + 1}", expanded=False):
                st.markdown(esc(step.get("critique", "")).replace("\n", "<br>"),
                            unsafe_allow_html=True)

        elif kind == "rewrite":
            st.markdown(f"""
<div class="trace-item active">
<div class="trace-title">Query rewrite — retry {retry}</div>
<div class="trace-sub">Previous retrieval insufficient. New query below.</div>
</div>""", unsafe_allow_html=True)
            with st.expander(f"Rewritten query — retry {retry}", expanded=True):
                st.markdown(f'<div class="qbox">{esc(step.get("query",""))}</div>',
                            unsafe_allow_html=True)


# ---- header ----

st.markdown('<div class="label">Retrieval-Augmented Generation</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Self-Healing RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Upload documents. Ask questions. The pipeline self-corrects when retrieval is weak.</div>', unsafe_allow_html=True)


# ---- knowledge base ----

rule()
label("Knowledge base")

st.markdown("""
<div class="paper">
<div class="paper-title">Add documents</div>
<div class="paper-sub">
PDF (text or scanned), DOCX, or TXT.
Text is extracted, chunked, embedded with qwen3-embedding:0.6b,
and stored in ChromaDB. Duplicates are skipped automatically.
</div>
</div>""", unsafe_allow_html=True)

# no hard Streamlit upload cap — set to None to rely only on server.maxUploadSize
uploaded_files = st.file_uploader(
    "Upload",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

col_btn, col_hint = st.columns([1, 2])
with col_btn:
    index_button = st.button("Add to knowledge base", use_container_width=True)
with col_hint:
    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) selected")
    else:
        st.caption("PDF · DOCX · TXT — set server.maxUploadSize in config.toml for files > 200 MB")


# ---- indexing ----

if index_button:
    if not uploaded_files:
        st.warning("Select at least one file.")
    else:
        total_new = 0
        processed = []
        failed = []
        progress = st.progress(0)
        status = st.empty()

        for i, f in enumerate(uploaded_files):
            try:
                status.info(f"Processing {f.name}...")
                docs = extract_documents(f)
                if not docs:
                    failed.append((f.name, "No extractable text"))
                    progress.progress((i + 1) / len(uploaded_files))
                    continue
                chunks = split_documents(docs)
                n = add_documents(chunks)
                total_new += n
                processed.append((f.name, len(chunks), n))
            except Exception as e:
                failed.append((f.name, str(e)))
            progress.progress((i + 1) / len(uploaded_files))

        status.empty()

        if total_new > 0:
            st.success(f"Added {total_new} new chunk(s).")
        elif processed:
            st.info("Files already indexed — no duplicates added.")
        for fname, err in failed:
            st.error(f"{fname}: {err}")

        st.rerun()


# ---- index status ----

try:
    current_count = count_documents()
except Exception:
    current_count = 0

rule()
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(metric("Indexed chunks", current_count, current_count > 0), unsafe_allow_html=True)
with c2:
    st.markdown(metric("Embedding model", "qwen3-embedding:0.6b"), unsafe_allow_html=True)
with c3:
    st.markdown(metric("Answer model", "qwen2.5-coder:7b"), unsafe_allow_html=True)


# ---- question ----

rule()
label("Ask your knowledge base")

question = st.text_area(
    "Question",
    placeholder="Ask something about the documents you uploaded...",
    height=110,
    label_visibility="collapsed"
)

ask_button = st.button("Ask", use_container_width=True)


# ---- run rag ----

if ask_button:
    if not question.strip():
        st.warning("Enter a question.")
    elif current_count == 0:
        st.warning("Knowledge base is empty — upload and index a document first.")
    else:
        with st.spinner("Running..."):
            try:
                result = graph.invoke({
                    "original_question": question.strip(),
                    "retrieval_query": question.strip(),
                    "documents": [],
                    "retrieval_results": [],
                    "answer": "",
                    "critique": "",
                    "grounded": False,
                    "sufficient_context": False,
                    "retry_count": 0,
                    "trace": []
                })

                if not result.get("grounded", False):
                    result["answer"] = (
                        "Couldn't find enough information in the uploaded documents "
                        "to answer this question reliably."
                    )

                st.session_state["rag_result"] = result
                st.session_state["last_question"] = question.strip()

            except Exception as e:
                st.error("Pipeline failed.")
                with st.expander("Error details"):
                    st.exception(e)


# ---- result ----

if "rag_result" in st.session_state:
    result = st.session_state["rag_result"]
    trace = result.get("trace", [])

    # answer
    rule()
    label("Answer")
    answer = result.get("answer", "No answer generated.")
    st.markdown(
        f'<div class="answer-block">{esc(answer).replace(chr(10), "<br>")}</div>',
        unsafe_allow_html=True
    )

    # status row
    rule()
    label("RAG status")
    grounded = result.get("grounded", False)
    sufficient = result.get("sufficient_context", False)
    retries = result.get("retry_count", 0)

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(metric("Grounded", "yes" if grounded else "no", grounded), unsafe_allow_html=True)
    with s2:
        st.markdown(metric("Context", "sufficient" if sufficient else "insufficient", sufficient), unsafe_allow_html=True)
    with s3:
        st.markdown(metric("Heal retries", retries, retries > 0), unsafe_allow_html=True)

    # last retrieval query
    retrieval_steps = [s for s in trace if s.get("step") == "retrieve"]
    if retrieval_steps:
        last = retrieval_steps[-1]
        rule()
        label("Final retrieval")
        st.markdown(f"""
<div class="paper">
<div class="paper-title">Query used</div>
<div class="qbox" style="margin-top:8px">{esc(last.get('query',''))}</div>
<div class="paper-sub" style="margin-top:8px">{len(last.get('results',[]))} chunks retrieved</div>
</div>""", unsafe_allow_html=True)

    # trace
    rule()
    render_trace(trace)

    # pipeline
    rule()
    label("Pipeline")
    healed = any(s.get("step") == "rewrite" for s in trace)
    heal_nodes = """
<div class="parrow">→</div>
<div class="pnode on">Rewrite</div>
<div class="parrow">→</div>
<div class="pnode on">Re-retrieve</div>
""" if healed else ""

    st.markdown(f"""
<div class="pipeline">
<div class="pnode on">Query</div>
<div class="parrow">→</div>
<div class="pnode on">Embed</div>
<div class="parrow">→</div>
<div class="pnode on">Retrieve</div>
<div class="parrow">→</div>
<div class="pnode on">qwen2.5-coder:7b</div>
<div class="parrow">→</div>
<div class="pnode on">Critic</div>
{heal_nodes}
</div>""", unsafe_allow_html=True)


# ---- footer ----

rule()
st.markdown("""
<div style="text-align:center;font-family:'IBM Plex Mono',monospace;font-size:9px;color:#2a2826;padding:8px">
CHROMADB · QWEN3-EMBEDDING:0.6B · QWEN2.5-CODER:7B · LANGGRAPH
</div>""", unsafe_allow_html=True)
