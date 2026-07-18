"""Prompt templates as Jinja2 strings.

Templates are versioned: prompt_version is part of the experiment hash.
Editing a template's text requires bumping its version key here so old
experiments aren't silently invalidated.
"""

from __future__ import annotations

from jinja2 import Environment, StrictUndefined

env = Environment(undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)


# --- Question generation (QuOTE-style) ---

QUESTION_GEN_BASIC_V1 = """Read the following text and generate {{ n }} factual question-answer pairs that capture the most important information.
Format each pair on its own line as: Question? Answer
Do not number the pairs. Do not use bullets. Do not include any preamble.

Text:
{{ chunk_text }}
"""

QUESTION_GEN_COMPLEX_V1 = """Read the following text and generate {{ n }} factual question-answer pairs designed to resemble authentic user search queries.
Each question should:
- accurately and semantically capture an important aspect of the text;
- vary in length and complexity (some short keyword queries, some natural language);
- include a mix of who/what/when/where/how/why questions;
- avoid pronouns: use the actual names or entities;
- avoid phrases like "according to the text" or "in the passage".

Format each question-answer pair on a single line as: Question? Answer
No numbering, no bullets, no commentary.

Text:
{{ chunk_text }}
"""

QUESTION_GEN_MULTIHOP_V1 = """Read the following text and generate {{ n }} multi-hop question-answer pairs.
Each question should require integrating multiple pieces of information from the text to answer.
- Use specific entity names, never pronouns.
- Avoid "according to the text" framing.
- Each question must be one sentence.

Format each pair on a single line as: Question? Answer

Text:
{{ chunk_text }}
"""


# --- Answer generation for triplet building ---

ANSWER_GEN_RAG_DEFAULT_V1 = """You are a helpful assistant. Use ONLY the information in the contexts below to answer the question.
If the answer is not contained in the contexts, say "I don't know."
Keep your answer concise.

{% for c in contexts %}
Context [{{ loop.index }}]:
{{ c }}

{% endfor %}
Question: {{ question }}

Answer:"""


# --- Inference prompts ---

VANILLA_RAG_V1 = """You are a helpful assistant. Use ONLY the information in the contexts below to answer the question.
If the answer is not contained in the contexts, say "I don't know."
Keep your answer concise.

{% for c in contexts %}
Context [{{ loop.index }}]:
{{ c }}

{% endfor %}
Question: {{ question }}

Answer:"""


TRIPLET_RAG_V1 = """You are a helpful assistant. Below are example question-answer pairs along with the contexts they were derived from.
Use these examples to understand the kind of reasoning required, then answer the final question using the contexts.
Answer ONLY from the contexts provided. If the answer is not present, say "I don't know."
Keep your answer concise.

{% for t in triplets %}
Example {{ loop.index }}:
Contexts:
{% for c in t.contexts %}- {{ c }}
{% endfor %}
Question: {{ t.question }}
Answer: {{ t.answer }}

{% endfor %}
{% if fresh_contexts %}
Additional contexts for the actual question:
{% for c in fresh_contexts %}
Context [{{ loop.index }}]:
{{ c }}

{% endfor %}
{% endif %}
Now answer this question:
Question: {{ question }}

Answer:"""


QA_DEMO_RAG_V1 = """You are a helpful assistant. Below are example question-answer pairs.
Use them to understand the answering style, then answer the final question.

{% for t in triplets %}
Example {{ loop.index }}:
Question: {{ t.question }}
Answer: {{ t.answer }}

{% endfor %}
{% if fresh_contexts %}
Contexts for the question:
{% for c in fresh_contexts %}
Context [{{ loop.index }}]:
{{ c }}

{% endfor %}
{% endif %}
Question: {{ question }}

Answer:"""


# --- Registry ---

PROMPTS: dict[str, str] = {
    "question_gen.basic.v1": QUESTION_GEN_BASIC_V1,
    "question_gen.complex.v1": QUESTION_GEN_COMPLEX_V1,
    "question_gen.multihop.v1": QUESTION_GEN_MULTIHOP_V1,
    "answer_gen.rag_default.v1": ANSWER_GEN_RAG_DEFAULT_V1,
    "infer.vanilla_rag.v1": VANILLA_RAG_V1,
    "infer.triplet_rag.v1": TRIPLET_RAG_V1,
    "infer.qa_demo_rag.v1": QA_DEMO_RAG_V1,
}


def render(key: str, **kwargs: object) -> str:
    if key not in PROMPTS:
        raise KeyError(f"Unknown prompt key: {key}. Known: {list(PROMPTS)}")
    template = env.from_string(PROMPTS[key])
    return template.render(**kwargs)


def question_gen_key(prompt_choice: str, version: str = "v1") -> str:
    return f"question_gen.{prompt_choice}.{version}"


def answer_gen_key(prompt_choice: str, version: str = "v1") -> str:
    return f"answer_gen.{prompt_choice}.{version}"


def infer_key(strategy: str, version: str = "v1") -> str:
    return f"infer.{strategy}.{version}"
