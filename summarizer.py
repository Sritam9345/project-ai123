import fitz
import re

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import requests
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import httpx
import tiktoken

from model import chatmodel as client

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

async def extract_pdf_text(pdf) -> str:
    
    pdf_bytes = await pdf.read()

    doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    text = []
    for page in doc:
        text.append(page.get_text())

    doc.close()

    return "\n".join(text)

def regex_mask(text: str) -> tuple[str, dict]:
    """Mask PII via regex and return (masked_text, placeholder→original mapping)."""

    mapping: dict[str, str] = {}
    counters: dict[str, int] = {}

    patterns = {
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "PHONE": r"\b(?:\+91[- ]?)?[6-9]\d{9}\b",
        "AADHAAR": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
        "PAN": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
        "CREDIT_CARD": r"\b(?:\d{4}[- ]?){3}\d{4}\b",
    }

    for label, pattern in patterns.items():
        counters[label] = 0

        def _replacer(m, _label=label):
            counters[_label] += 1
            placeholder = f"[{_label}_MASKED_{counters[_label]}]"
            mapping[placeholder] = m.group(0)
            return placeholder

        text = re.sub(pattern, _replacer, text)

    return text, mapping

def ner_mask(text: str, mapping: dict) -> str:
    """Mask NER-detected entities and extend the placeholder→original mapping."""

    entities = analyzer.analyze(
        text=text,
        language="en"
    )

    # Sort descending by position so replacements don't shift earlier spans
    entities_sorted = sorted(entities, key=lambda e: e.start, reverse=True)

    counters: dict[str, int] = {}
    for entity in entities_sorted:
        entity_type = entity.entity_type
        original_value = text[entity.start:entity.end]

        counters[entity_type] = counters.get(entity_type, 0) + 1
        placeholder = f"[{entity_type}_MASKED_{counters[entity_type]}]"

        mapping[placeholder] = original_value
        text = text[: entity.start] + placeholder + text[entity.end :]

    return text



def unmask(text: str, mapping: dict) -> str:
    """Restore all placeholders in *text* to their original PII values."""
    for placeholder, original in mapping.items():
        text = text.replace(placeholder, original)
    return text


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks



async def process_pdf(pdf):

    text = await extract_pdf_text(pdf)

    text, mapping = regex_mask(text)

    text = ner_mask(text, mapping)

    chunks = chunk_text(
        text,
        chunk_size=1000,
        overlap=200
    )

    return chunks, mapping




async def selectedChunks(chunks):

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/generate",
            json={"chunks": chunks}
        )

    embeddings = np.array(
        response.json()["embeddings"]
    )

    centroid = np.mean(
        embeddings,
        axis=0
    ).reshape(1, -1)

    scores = cosine_similarity(
        embeddings,
        centroid
    ).flatten()

    keep_count = max(
        1,
        int(len(chunks) * 0.3)
    )

    top_indices = np.argsort(scores)[-keep_count:]

    selected_chunks = [
        chunks[i]
        for i in sorted(top_indices)
    ]

    return selected_chunks

_PLACEHOLDER_RE = re.compile(r'\[[A-Z_]+_MASKED_\d+\]')


def _extract_placeholders(text: str) -> list[str]:
    """Return ordered unique placeholders found in *text*."""
    seen: dict[str, None] = {}          # dict preserves insertion order, deduplicates
    for p in _PLACEHOLDER_RE.findall(text):
        seen[p] = None
    return list(seen)


def _correction_pass(summary: str, missing: set[str]) -> str:
    """Ask the model to reintegrate placeholders it dropped in the first pass."""
    missing_list = "\n".join(f"  {p}" for p in sorted(missing))

    correction_prompt = f"""The summary below is missing PII placeholders that were present in the \
source document. Each placeholder stands for a real value that will be restored later, \
so losing one means data loss.

MISSING PLACEHOLDERS (every one must appear verbatim in your output):
{missing_list}

RULES:
- Insert each missing placeholder at the most contextually appropriate location.
- Do NOT alter any other sentence, word, or existing placeholder.
- Do NOT add commentary or explanations.
- Return only the corrected summary.

SUMMARY:
{summary}"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise text editor. Your only job is to restore "
                    "missing placeholders into a summary without changing anything else."
                )
            },
            {
                "role": "user",
                "content": correction_prompt
            }
        ],
        temperature=0
    )
    return response.choices[0].message.content


def summarize_chunks(chunks):

    document = "\n".join(chunks)

    enc = tiktoken.get_encoding("cl100k_base")
    print(len(enc.encode(document)))
    print(document)

    # Build the explicit checklist so the model knows exactly what must survive
    placeholders = _extract_placeholders(document)
    placeholder_checklist = "\n".join(f"  {p}" for p in placeholders) or "  (none)"

    prompt = f"""<task>
Summarize the document provided in <document> tags.
</task>

<hard_constraints>
PLACEHOLDER PRESERVATION - THIS IS YOUR TOP PRIORITY:
The document contains masked PII placeholders. Every placeholder listed in
<required_placeholders> MUST appear in your summary verbatim and unchanged.

- Copy each placeholder character-for-character, including brackets and numbers.
- Never paraphrase, rename, merge, drop, or describe a placeholder in prose
  (e.g. do NOT write "their email" instead of [EMAIL_MASKED_1]).
- If a person or entity is referred to only by a placeholder, use that placeholder
  wherever you would normally use their name.
- After writing your summary, mentally scan for every placeholder in
  <required_placeholders> and confirm each one is present before finishing.
</hard_constraints>

<required_placeholders>
{placeholder_checklist}
</required_placeholders>

<summarization_requirements>
- Preserve all key information, findings, conclusions, and relationships.
- Remove redundancy and repetitive details.
- Maintain factual accuracy; do NOT invent information.
- Generate a detailed summary of approximately 3500 words.
- Use concise, professional language.
</summarization_requirements>

<document>
{document}
</document>"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert document summarization assistant. "
                    "You treat placeholder preservation as a hard constraint that "
                    "overrides all stylistic preferences. A summary with a missing "
                    "placeholder is considered incorrect output."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    summary = response.choices[0].message.content

    # Verify every placeholder survived; run a targeted correction pass if not
    required = set(placeholders)
    present  = set(_extract_placeholders(summary))
    missing  = required - present

    if missing:
        print(f"[summarizer] correction pass triggered for: {missing}")
        summary = _correction_pass(summary, missing)

    return summary

    

async def generateResponse(pdf):

    chunks, mapping = await process_pdf(pdf)

    selected_chunks = await selectedChunks(chunks)

    response = summarize_chunks(selected_chunks)

    return unmask(response, mapping)