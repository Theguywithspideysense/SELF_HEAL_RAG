import re
from typing import TypedDict, List

from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from src.vectorstore import get_vectorstore
from src.rag import generate_answer


TOP_K = 10
MAX_RETRIES = 3
GROUNDING_THRESHOLD = 0.25
RELEVANCE_THRESHOLD = 0.20

_STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "your", "with", "this",
    "that", "have", "has", "had", "was", "were", "will", "would", "could",
    "should", "can", "does", "did", "from", "into", "about", "which", "what",
    "when", "where", "who", "whom", "why", "how", "there", "here", "their",
    "them", "they", "these", "those", "then", "than", "also", "such", "some",
    "any", "all", "each", "over", "under", "above", "below", "between",
    "being", "been", "context", "document", "documents", "answer", "question"
}


def _keywords(text: str):
    words = re.findall(r"[A-Za-z0-9]{4,}", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def _coverage_ratio(text: str, context_lower: str):
    keywords = _keywords(text)
    if not keywords:
        return 1.0, []
    found = [w for w in keywords if w in context_lower]
    missing = [w for w in keywords if w not in context_lower]
    return len(found) / len(keywords), missing[:10]


class GraphState(TypedDict):
    original_question: str
    retrieval_query: str
    documents: List
    retrieval_results: List
    answer: str
    critique: str
    grounded: bool
    sufficient_context: bool
    retry_count: int
    trace: List


critic_llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0.1)


def retrieve(state: GraphState):
    print("\n[RETRIEVAL]")
    query = state["retrieval_query"]
    print(f"Query: {query}")

    try:
        results = get_vectorstore().similarity_search_with_relevance_scores(query, k=TOP_K)
    except Exception as e:
        print(f"Retrieval error: {e}")
        results = []

    documents, retrieval_results = [], []
    for rank, (doc, score) in enumerate(results, start=1):
        documents.append(doc)
        info = {
            "rank": rank,
            "score": round(float(score), 4),
            "source": doc.metadata.get("source", "Unknown"),
            "preview": doc.page_content.replace("\n", " ")[:300]
        }
        if doc.metadata.get("page") is not None:
            info["page"] = doc.metadata["page"]
        retrieval_results.append(info)
        print(f"[{rank}] {info['source']} score={info['score']}")

    trace = state.get("trace", []).copy()
    trace.append({
        "step": "retrieve",
        "retry": state["retry_count"],
        "query": query,
        "chunks": len(documents),
        "results": retrieval_results
    })

    return {"documents": documents, "retrieval_results": retrieval_results, "trace": trace}


def generate(state: GraphState):
    print("\n[GENERATE]")
    answer = generate_answer(state["original_question"], state["documents"])
    print(f"Answer:\n{answer}")

    trace = state.get("trace", []).copy()
    trace.append({"step": "generate", "retry": state["retry_count"]})
    return {"answer": answer, "trace": trace}


def critic(state: GraphState):
    print("\n[CRITIC]")
    documents = state["documents"]
    question = state["original_question"]
    answer = state["answer"]

    if not documents:
        trace = state.get("trace", []).copy()
        trace.append({"step": "critic", "retry": state["retry_count"],
                      "grounded": False, "sufficient_context": False,
                      "critique": "No evidence retrieved."})
        return {"critique": "No evidence retrieved.", "grounded": False,
                "sufficient_context": False, "trace": trace}

    context = "\n\n".join(d.page_content for d in documents)
    context_lower = context.lower()

    prompt = f"""You are an evaluator for a RAG system.

QUESTION:
{question}

RETRIEVED EVIDENCE:
{context}

GENERATED ANSWER:
{answer}

Evaluate whether the answer is reasonably supported by the evidence.
Paraphrasing is fine. Combining chunks is fine. Be lenient.
Only REVISE if an important claim is clearly unsupported or contradictory.

Return ONLY:
DECISION: ACCEPT
or
DECISION: REVISE

Then one short reason.
"""

    critique = str(critic_llm.invoke(prompt).content).strip()
    print(f"Critic:\n{critique}")

    match = re.search(r"(?i)DECISION\s*:\s*(ACCEPT|REVISE)", critique)
    decision = match.group(1).upper() if match else None

    answer_coverage, missing = _coverage_ratio(answer, context_lower)
    question_coverage, _ = _coverage_ratio(question, context_lower)
    sufficient_context = question_coverage >= RELEVANCE_THRESHOLD

    if not sufficient_context:
        grounded = False
    elif decision == "ACCEPT":
        grounded = True
    elif decision == "REVISE":
        grounded = answer_coverage >= GROUNDING_THRESHOLD
    else:
        grounded = answer_coverage >= GROUNDING_THRESHOLD

    if sufficient_context and answer_coverage >= 0.35:
        grounded = True

    final_critique = (
        f"ACCEPTED. Coverage: {answer_coverage:.0%}. Answer is supported by evidence."
        if grounded else
        f"REVISION NEEDED. Coverage: {answer_coverage:.0%}. Evidence does not adequately support the answer."
    )

    print(f"Grounded: {grounded} | Context sufficient: {sufficient_context} | Coverage: {answer_coverage:.0%}")

    trace = state.get("trace", []).copy()
    trace.append({
        "step": "critic",
        "retry": state["retry_count"],
        "grounded": grounded,
        "sufficient_context": sufficient_context,
        "coverage_ratio": round(answer_coverage, 3),
        "missing_keywords": missing,
        "critique": final_critique
    })

    return {"critique": final_critique, "grounded": grounded,
            "sufficient_context": sufficient_context, "trace": trace}


def rewrite_query(state: GraphState):
    print("\n[REWRITE]")
    prompt = f"""You are a search query optimizer for a document QA system.

USER QUESTION: {state['original_question']}
CURRENT QUERY: {state['retrieval_query']}
EVALUATION: {state['critique']}

Write a better semantic search query.
Rules: preserve meaning, add synonyms, stay concise, return ONLY the query.

NEW QUERY:"""

    new_query = str(critic_llm.invoke(prompt).content).strip()
    new_query = re.sub(r"^(NEW QUERY|QUERY|Query)\s*:\s*", "", new_query, flags=re.I).strip().strip('"').strip("'")

    if not new_query or new_query.lower() == state["retrieval_query"].lower():
        new_query = f"{state['retrieval_query']} key concepts explanation"

    print(f"Old: {state['retrieval_query']}\nNew: {new_query}")

    new_retry = state["retry_count"] + 1
    trace = state.get("trace", []).copy()
    trace.append({"step": "rewrite", "retry": new_retry, "query": new_query})

    return {"retrieval_query": new_query, "retry_count": new_retry, "trace": trace}


def decide_next_step(state: GraphState):
    if state["grounded"]:
        print("\n[ACCEPTED]")
        return "end"
    if state["retry_count"] >= MAX_RETRIES:
        print("\n[MAX RETRIES]")
        return "end"
    print("\n[HEALING]")
    return "rewrite"


workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("critic", critic)
workflow.add_node("rewrite", rewrite_query)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "critic")
workflow.add_conditional_edges("critic", decide_next_step, {"end": END, "rewrite": "rewrite"})
workflow.add_edge("rewrite", "retrieve")

graph = workflow.compile()
print("\n[RAG GRAPH READY]")