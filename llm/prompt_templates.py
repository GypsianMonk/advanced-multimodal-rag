"""
llm/prompt_templates.py
────────────────────────
Centralised prompt template registry.
Templates are intent-aware — different prompts for different query types.
"""

from __future__ import annotations

from string import Template

# ─── Base Templates ───────────────────────────────────────────────────────────

FACTUAL = Template("""You are a precise, knowledge-grounded assistant.
Answer the question using ONLY the provided context. Be concise and factual.
If the context is insufficient, say: "The provided documents do not contain enough information."

CONTEXT:
$context

QUESTION: $query

ANSWER:""")

ANALYTICAL = Template("""You are an expert analyst.
Using the provided context, provide a thorough analysis of the question.
Structure your response with: Key Findings, Supporting Evidence, and Conclusion.
Cite specific sources where possible.

CONTEXT:
$context

ANALYTICAL QUESTION: $query

ANALYSIS:""")

COMPARATIVE = Template("""You are a detail-oriented research assistant.
Compare and contrast based on the provided context only.
Use a structured format: present both sides clearly before summarising.

CONTEXT:
$context

COMPARISON QUESTION: $query

COMPARISON:""")

VISUAL = Template("""You are an expert at interpreting visual data and charts.
The following context describes visual content (charts, graphs, images).
Describe and interpret the visual information to answer the question.

CONTEXT:
$context

VISUAL QUESTION: $query

INTERPRETATION:""")

SUMMARY = Template("""You are a concise summariser.
Summarise the key points from the provided context relevant to the topic.
Use bullet points. Limit to 5–7 key takeaways.

CONTEXT:
$context

TOPIC: $query

SUMMARY:""")

# ─── Template Registry ────────────────────────────────────────────────────────

TEMPLATE_MAP: dict[str, Template] = {
    "factual": FACTUAL,
    "analytical": ANALYTICAL,
    "comparative": COMPARATIVE,
    "visual": VISUAL,
    "summary": SUMMARY,
}

DEFAULT_TEMPLATE = FACTUAL


def get_prompt(intent: str, query: str, context: str) -> str:
    """Render the appropriate prompt template for a given intent."""
    template = TEMPLATE_MAP.get(intent, DEFAULT_TEMPLATE)
    return template.substitute(query=query, context=context)
