from langchain_ollama import ChatOllama


MODEL_NAME = "qwen2.5-coder:7b"


def create_llm():
    return ChatOllama(model=MODEL_NAME, temperature=0.1)


def build_context(documents):
    if not documents:
        return ""
    parts = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page")
        label = f"{source} | Page {page}" if page else source
        content = doc.page_content.strip()
        if content:
            parts.append(f"[EVIDENCE {i}]\nSOURCE: {label}\n\n{content}")
    return "\n\n".join(parts)


def clean_answer(answer):
    if not answer:
        return "• Couldn't generate an answer from the provided documents."

    answer = answer.strip()

    for prefix in ["ANSWER:", "Answer:", "FINAL ANSWER:", "Final Answer:"]:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()

    lines = answer.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            line = line.lstrip("#").strip()
        if len(line) >= 2 and line[0].isdigit() and line[1] in [".", ")"]:
            line = "• " + line[2:].strip()
        elif line.startswith("- "):
            line = "• " + line[2:]
        elif line.startswith("* "):
            line = "• " + line[2:]
        cleaned.append(line)

    if not cleaned:
        return "• Couldn't generate a useful answer from the provided documents."

    final = []
    for line in cleaned:
        if line.startswith("•"):
            final.append(line)
        elif line.endswith(":") and len(line) < 80:
            final.append(f"\n**{line}**")
        else:
            final.append(f"• {line}")

    return "\n".join(final)


def generate_answer(question, documents):
    if not documents:
        return "• No relevant information found in the uploaded documents."

    context = build_context(documents)

    prompt = f"""You are the answer generator in a Retrieval-Augmented Generation system.
Answer the user's question using ONLY the evidence below.

USER QUESTION
{question}

RETRIEVED EVIDENCE
{context}

RULES
1. Use retrieved evidence as the only source of truth.
2. Answer directly. Do not invent facts not in the evidence.
3. Combine information from multiple chunks when useful.
4. Paraphrase naturally. Do not copy large chunks verbatim.
5. If evidence only partially answers, say so clearly.
6. Avoid repeating the same idea twice.

OUTPUT FORMAT
Write a concise answer using bullet points (3-8 bullets).
If "how" is asked, use numbered steps.
If "why" is asked, list reasons as bullets.
If comparing, organize around differences.
Start with a short definition if defining something.
No long intro. No conclusion unless it adds value.

ANSWER
"""

    return clean_answer(create_llm().invoke(prompt).content)
