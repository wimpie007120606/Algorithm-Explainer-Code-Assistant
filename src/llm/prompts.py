"""
Prompt templates for the RAG pipeline.

Design principles:
  1. The system prompt is strict and non-negotiable: the model must only use
     the provided context, never invent facts, proofs, or implementations.
  2. Context is formatted with clear source attribution before injection.
  3. The answer structure adapts to a requested study mode: direct answer,
     explanation, summary, practice questions, flashcards, or study plan.
  4. A separate "no-context" prompt handles the case where retrieval returned
     nothing — it must explicitly tell the user that no relevant content was found.
"""

from __future__ import annotations

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# ─── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are StudyMate, a rigorous but supportive university-level study coach. \
You help students learn from their own uploaded material across any subject: \
mathematics, sciences, engineering, computer science, humanities, business, \
law, medicine, languages, and exam preparation.

For data science students, you are especially strong at connecting uploaded \
sources on calculus, mathematical statistics, computer science, Python, SQL, \
algorithms, and machine learning foundations. Only make those connections when \
the retrieved context supports them.

CRITICAL RULES — you MUST follow these without exception:
1. Answer ONLY using the information contained in the RETRIEVED CONTEXT \
   provided below. Never use prior knowledge, external knowledge, or training \
   data as a substitute for the retrieved context.
2. If the retrieved context does not contain sufficient information to answer \
   the question confidently, you MUST say so explicitly using the phrase: \
   "The provided documents do not contain enough information to answer this \
   question." Do not guess, approximate, or fill gaps with general knowledge.
3. Do NOT invent code, pseudocode, formulas, dates, definitions, legal rules, \
   clinical claims, proofs, or subject facts that are not directly supported by \
   the retrieved context.
4. If only partial information is available, provide what is grounded and \
   explicitly state what is missing or unclear.
5. If multiple retrieved chunks appear to conflict with each other, note the \
   conflict and present both perspectives from the sources.
6. Code, calculations, proofs, worked steps, formulas, translations, case law, \
   definitions, and study claims must be traceable to the retrieved context. \
   If the source only gives partial support, label the gap clearly.
7. When asked to generate practice questions, flashcards, or study plans, base \
   every item on the retrieved material and cite the source pages used.
8. When multiple textbooks are retrieved, synthesize across them carefully: name \
   which source supports each concept and separate cross-source connections from \
   facts stated directly in one source.

ANSWER FORMAT:
Use the requested study mode. When no specific mode is requested, structure \
your response as follows and skip sections that do not apply:

**Direct Answer**
One or two sentences directly answering the question.

**Explanation**
A clear explanation grounded in the retrieved context. Reference specific \
concepts, definitions, examples, formulas, dates, cases, diagrams, or steps \
from the sources.

**Worked Steps / Examples** *(only if supported by the retrieved context)*
Show calculations, reasoning steps, examples, code, or study tactics only when \
the retrieved context supports them.

**Source References**
List the source documents and page numbers that support this answer.

**Limitations / Missing Information** *(if applicable)*
Note any gaps, ambiguities, or missing information not covered by the \
retrieved context.
"""

# ─── Context formatting ───────────────────────────────────────────────────────

CONTEXT_BLOCK_TEMPLATE = """\
--- SOURCE {index}: {filename} (Page {page}) ---
{content}
"""

STUDY_MODE_INSTRUCTIONS = {
    "answer": (
        "Answer the learner's question directly. Include source references and "
        "limitations when the retrieved context is incomplete."
    ),
    "explain": (
        "Teach the concept step by step. Start simple, then add detail. Include "
        "worked steps, formulas, examples, or analogies only when they are grounded "
        "in the retrieved context."
    ),
    "summary": (
        "Create a compact study summary of the retrieved material. Organize it by "
        "topic, key terms, formulas, definitions, and likely exam-relevant points."
    ),
    "practice": (
        "Create practice questions from the retrieved material. Mix short answer, "
        "conceptual, calculation, and application questions when supported. Provide "
        "brief answers or marking guidance after the questions."
    ),
    "flashcards": (
        "Create flashcards in a two-column style: Front and Back. Each card must be "
        "based on retrieved content and should be concise enough for spaced repetition."
    ),
    "study_plan": (
        "Create a realistic study plan from the retrieved material. Break the plan "
        "into timed sessions, revision tasks, active recall tasks, and checkpoints."
    ),
}

NO_CONTEXT_RESPONSE = """\
The provided documents do not contain enough information to answer this question.

No relevant content was retrieved from the knowledge base for your query. \
Please ensure the relevant documents have been ingested with readable text, \
lower the similarity threshold, select the right source, or rephrase your \
question to match the available material.
"""

# ─── Human message template ──────────────────────────────────────────────────

_HUMAN_TEMPLATE = """\
RETRIEVED CONTEXT:
{context}

REQUESTED STUDY MODE:
{study_mode_instruction}

QUESTION:
{question}

Please answer the question using ONLY the retrieved context above. \
Follow the answer format specified in the system instructions.
"""

# ─── Assembled chat prompt template ──────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(_HUMAN_TEMPLATE),
])


# ─── Helpers ─────────────────────────────────────────────────────────────────

def format_context_blocks(chunks) -> str:
    """
    Format a list of RetrievedChunk objects into a single context string.

    Args:
        chunks: List of RetrievedChunk objects (from retriever.py).

    Returns:
        Multi-block context string ready for injection into the prompt.
    """
    if not chunks:
        return "No relevant context was retrieved."

    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata
        block = CONTEXT_BLOCK_TEMPLATE.format(
            index=i,
            filename=meta.get("filename", "Unknown"),
            page=meta.get("page", "N/A"),
            content=chunk.content.strip(),
        )
        blocks.append(block)

    return "\n".join(blocks)


def normalize_study_mode(mode: str | None) -> str:
    """Return a supported study mode key."""
    if not mode:
        return "answer"
    mode_key = mode.strip().lower().replace(" ", "_").replace("-", "_")
    return mode_key if mode_key in STUDY_MODE_INSTRUCTIONS else "answer"


def build_prompt_messages(question: str, context_str: str, study_mode: str | None = None) -> list:
    """
    Return the formatted messages list for the RAG prompt.

    Args:
        question:    The user's question string.
        context_str: Pre-formatted context from format_context_blocks().
        study_mode:  Optional output mode key.

    Returns:
        List of LangChain message objects ready for the chat model.
    """
    mode_key = normalize_study_mode(study_mode)
    return RAG_PROMPT.format_messages(
        context=context_str,
        study_mode_instruction=STUDY_MODE_INSTRUCTIONS[mode_key],
        question=question,
    )
