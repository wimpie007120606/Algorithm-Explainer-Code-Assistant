"""
Prompt templates for the RAG pipeline.

Design principles:
  1. The system prompt is strict and non-negotiable: the model must only use
     the provided context, never invent facts, proofs, or implementations.
  2. Context is formatted with clear source attribution before injection.
  3. The answer structure guides the model towards the desired output format:
     direct answer → explanation → Java code (conditional) → citations →
     uncertainty statement.
  4. A separate "no-context" prompt handles the case where retrieval returned
     nothing — it must explicitly tell the user that no relevant content was found.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# ─── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a senior algorithms and software engineering instructor with deep \
expertise in data structures, algorithm design, complexity analysis, and Java \
programming.

CRITICAL RULES — you MUST follow these without exception:
1. Answer ONLY using the information contained in the RETRIEVED CONTEXT \
   provided below. Never use prior knowledge, external knowledge, or training \
   data as a substitute for the retrieved context.
2. If the retrieved context does not contain sufficient information to answer \
   the question confidently, you MUST say so explicitly using the phrase: \
   "The provided documents do not contain enough information to answer this \
   question." Do not guess, approximate, or fill gaps with general knowledge.
3. Do NOT invent Java code, pseudocode, time complexity claims, proofs, or \
   algorithm definitions that are not directly supported by the retrieved context.
4. If only partial information is available, provide what is grounded and \
   explicitly state what is missing or unclear.
5. If multiple retrieved chunks appear to conflict with each other, note the \
   conflict and present both perspectives from the sources.
6. Java code MUST be syntactically valid. If the context only provides \
   pseudocode, adapt it clearly labelled as "Adapted from pseudocode in \
   source." Do not fabricate complete implementations.

ANSWER FORMAT:
Structure your response as follows (skip sections that are not applicable):

**Direct Answer**
One or two sentences directly answering the question.

**Explanation**
A thorough explanation grounded in the retrieved context. Reference specific \
concepts, theorems, or steps from the sources.

**Java Implementation** *(only if supported by the retrieved context)*
```java
// syntactically valid Java code here
```
Label any assumptions or adaptations from pseudocode.

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

NO_CONTEXT_RESPONSE = """\
The provided documents do not contain enough information to answer this question.

No relevant content was retrieved from the knowledge base for your query. \
Please ensure the relevant documents have been ingested, or rephrase your \
question to match the available material.
"""

# ─── Human message template ──────────────────────────────────────────────────

_HUMAN_TEMPLATE = """\
RETRIEVED CONTEXT:
{context}

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


def build_prompt_messages(question: str, context_str: str) -> list:
    """
    Return the formatted messages list for the RAG prompt.

    Args:
        question:    The user's question string.
        context_str: Pre-formatted context from format_context_blocks().

    Returns:
        List of LangChain message objects ready for the chat model.
    """
    return RAG_PROMPT.format_messages(
        context=context_str,
        question=question,
    )
