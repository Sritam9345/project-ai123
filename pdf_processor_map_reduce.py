from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable

import fitz
import tiktoken
from faker import Faker
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer import RecognizerResult

from model import chatmodel as client


analyzer = AnalyzerEngine()
faker = Faker()


# ---------- PDF extraction ----------

async def extract_pdf_text(pdf) -> str:
    """
    Read a PDF upload and return the full extracted text.
    """
    pdf_bytes = await pdf.read()

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: list[str] = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()

    return "\n".join(pages)


# ---------- Masking / unmasking ----------

_ENTITY_TOKEN_RE = re.compile(r"\[[A-Z][A-Z0-9_]*_[A-Za-z0-9_]+_[0-9a-f]{8}\]")


@dataclass
class MaskingResult:
    masked_text: str
    mapping: dict[str, str]


def _safe_name_part(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "X"


def _fake_token(entity_type: str) -> str:
    """
    Generate a unique fake token that looks like:
    [PERSON_Alice_Smith_ab12cd34]
    """
    first = _safe_name_part(faker.first_name())
    last = _safe_name_part(faker.last_name())
    short_uuid = uuid.uuid4().hex[:8]
    return f"[{entity_type.upper()}_{first}_{last}_{short_uuid}]"


def _non_overlapping_results(results: list[RecognizerResult]) -> list[RecognizerResult]:
    """
    Keep the highest-priority non-overlapping entities.
    Presidio can return overlapping spans; we keep the earliest, longest ones.
    """
    ordered = sorted(
        results,
        key=lambda r: (
            r.start,
            -(r.end - r.start),
            -float(r.score or 0.0),
        ),
    )

    chosen: list[RecognizerResult] = []
    last_end = -1

    for result in ordered:
        if result.start >= last_end:
            chosen.append(result)
            last_end = result.end

    return chosen


def mask_with_presidio_and_faker(text: str) -> MaskingResult:
    """
    Detect entities with Presidio and replace each distinct entity text with a
    unique faker-based token. The mapping is returned so the summary can be
    unmasked later.
    """
    results = analyzer.analyze(text=text, language="en")
    results = _non_overlapping_results(results)

    # Reuse the same fake token for repeated occurrences of the same entity text.
    original_to_token: dict[tuple[str, str], str] = {}
    token_to_original: dict[str, str] = {}

    # Replace from the end so character positions remain valid.
    results = sorted(results, key=lambda r: r.start, reverse=True)

    for result in results:
        original_value = text[result.start : result.end]
        entity_type = (result.entity_type or "ENTITY").upper()
        key = (original_value, entity_type)

        if key not in original_to_token:
            token = _fake_token(entity_type)
            original_to_token[key] = token
            token_to_original[token] = original_value

        token = original_to_token[key]
        text = text[: result.start] + token + text[result.end :]

    return MaskingResult(masked_text=text, mapping=token_to_original)


def unmask(text: str, mapping: dict[str, str]) -> str:
    """
    Restore fake tokens back to the original entity strings.
    """
    for token, original in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        text = text.replace(token, original)
    return text


# ---------- Chunking / batching ----------

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    """
    Split the document into overlapping character chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap

    return chunks


def batch_items_by_chars(items: list[str], max_chars: int = 12000) -> list[list[str]]:
    """
    Group chunks into batches so each batch stays roughly within the size budget.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")

    batches: list[list[str]] = []
    current: list[str] = []
    current_size = 0

    for item in items:
        item_size = len(item)
        # If a single item is bigger than the budget, keep it alone.
        if item_size >= max_chars:
            if current:
                batches.append(current)
                current = []
                current_size = 0
            batches.append([item])
            continue

        if current and current_size + item_size > max_chars:
            batches.append(current)
            current = [item]
            current_size = item_size
        else:
            current.append(item)
            current_size += item_size

    if current:
        batches.append(current)

    return batches


# ---------- Prompt helpers ----------

def _extract_tokens(text: str) -> list[str]:
    """
    Collect unique fake tokens in the order they appear.
    """
    seen: dict[str, None] = {}
    for token in _ENTITY_TOKEN_RE.findall(text):
        seen[token] = None
    return list(seen.keys())


def _build_token_checklist(tokens: Iterable[str]) -> str:
    token_list = list(tokens)
    if not token_list:
        return "  (none)"
    return "\n".join(f"  {token}" for token in token_list)


def _summarize_with_prompt(document: str, required_tokens: list[str], system_prompt: str, user_intro: str, model: str = "gpt-4.1") -> str:
    """
    Generic summarization wrapper with hard token-preservation rules.
    """
    token_checklist = _build_token_checklist(required_tokens)

    prompt = f"""{user_intro}

<hard_constraints>
TOKEN PRESERVATION:
Every token listed in <required_tokens> must appear verbatim in the output.
Do not rename, paraphrase, merge, or drop any token.
If a token is present in the source, it must survive in the summary unchanged.
</hard_constraints>

<required_tokens>
{token_checklist}
</required_tokens>

<summary_requirements>
- Preserve the important facts, relationships, entities, and conclusions.
- Remove redundancy and repeated details.
- Do not invent information.
- Keep the output concise but useful.
</summary_requirements>

<document>
{document}
</document>"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


def _correction_pass(summary: str, missing_tokens: set[str], model: str = "gpt-4.1") -> str:
    """
    Ask the model to reinsert missing tokens that were dropped from the summary.
    """
    missing_list = "\n".join(f"  {token}" for token in sorted(missing_tokens))

    correction_prompt = f"""The summary below is missing tokens that were present in the source text.

MISSING TOKENS (every one must appear verbatim in your output):
{missing_list}

RULES:
- Insert each missing token at the most contextually appropriate location.
- Do NOT alter any other sentence, word, or existing token.
- Do NOT add commentary or explanations.
- Return only the corrected summary.

SUMMARY:
{summary}"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise text editor. Your only job is to restore "
                    "missing tokens into a summary without changing anything else."
                )
            },
            {"role": "user", "content": correction_prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


def _preserve_tokens(summary: str, required_tokens: list[str], model: str = "gpt-4.1") -> str:
    """
    Ensure every required token survived the generation step.
    """
    required = set(required_tokens)
    present = set(_extract_tokens(summary))
    missing = required - present

    if missing:
        summary = _correction_pass(summary, missing, model=model)

    return summary


# ---------- Map / reduce summarization ----------

def summarize_batch(batch_text: str, required_tokens: list[str], model: str = "gpt-4.1") -> str:
    system_prompt = (
        "You are an expert document summarization assistant. "
        "Treat token preservation as a hard constraint."
    )

    user_intro = """<task>
Summarize one batch from a larger PDF.
</task>

<instructions>
- Preserve the core facts from the batch.
- Keep the writing clean and professional.
- Do not use bullets unless they genuinely improve readability.
- Prefer neutral, faithful language.
</instructions>"""

    summary = _summarize_with_prompt(
        document=batch_text,
        required_tokens=required_tokens,
        system_prompt=system_prompt,
        user_intro=user_intro,
        model=model,
    )

    return _preserve_tokens(summary, required_tokens, model=model)


def reduce_summaries(batch_summaries: list[str], required_tokens: list[str], model: str = "gpt-4.1") -> str:
    combined = "\n\n".join(batch_summaries)

    system_prompt = (
        "You are an expert document summarization assistant. "
        "You combine partial summaries into one strong final summary. "
        "Token preservation is a hard constraint."
    )

    user_intro = """<task>
Merge multiple batch summaries into one final summary.
</task>

<instructions>
- Remove repetition across the batch summaries.
- Keep the most important ideas, outcomes, and relationships.
- Stay faithful to the source content.
- Write a polished final summary.
</instructions>"""

    summary = _summarize_with_prompt(
        document=combined,
        required_tokens=required_tokens,
        system_prompt=system_prompt,
        user_intro=user_intro,
        model=model,
    )

    return _preserve_tokens(summary, required_tokens, model=model)


# ---------- Full pipeline ----------

async def process_pdf(pdf, *, chunk_size: int = 1200, overlap: int = 150) -> tuple[str, MaskingResult, list[str]]:
    """
    Extract text, mask entities, and chunk the masked text.
    """
    text = await extract_pdf_text(pdf)
    masked = mask_with_presidio_and_faker(text)
    chunks = chunk_text(masked.masked_text, chunk_size=chunk_size, overlap=overlap)
    return text, masked, chunks


async def generate_response(
    pdf,
    *,
    chunk_size: int = 1200,
    overlap: int = 150,
    batch_max_chars: int = 12000,
    model: str = "gpt-4.1",
) -> dict[str, Any]:
    """
    End-to-end pipeline:
    1) extract text
    2) mask entities using Presidio + Faker
    3) chunk text
    4) map: summarize each batch
    5) reduce: merge all summaries into one final summary
    6) unmask the final summary
    7) return masked/unmasked summaries + time metrics
    """
    started = time.perf_counter()

    t0 = time.perf_counter()
    original_text = await extract_pdf_text(pdf)
    extract_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    masked_result = mask_with_presidio_and_faker(original_text)
    mask_seconds = time.perf_counter() - t1

    t2 = time.perf_counter()
    chunks = chunk_text(masked_result.masked_text, chunk_size=chunk_size, overlap=overlap)
    batch_groups = batch_items_by_chars(chunks, max_chars=batch_max_chars)
    chunk_seconds = time.perf_counter() - t2

    t3 = time.perf_counter()
    batch_summaries: list[str] = []
    for batch in batch_groups:
        batch_text = "\n\n".join(batch)
        batch_summary = summarize_batch(batch_text, required_tokens=_extract_tokens(batch_text), model=model)
        batch_summaries.append(batch_summary)
    map_seconds = time.perf_counter() - t3

    t4 = time.perf_counter()
    final_masked_summary = reduce_summaries(
        batch_summaries=batch_summaries,
        required_tokens=_extract_tokens("\n\n".join(batch_summaries)),
        model=model,
    )
    reduce_seconds = time.perf_counter() - t4

    t5 = time.perf_counter()
    final_unmasked_summary = unmask(final_masked_summary, masked_result.mapping)
    unmask_seconds = time.perf_counter() - t5

    total_seconds = time.perf_counter() - started

    return {
        "masked_summary": final_masked_summary,
        "unmasked_summary": final_unmasked_summary,
        "mapping": masked_result.mapping,
        "metrics": {
            "extract_seconds": round(extract_seconds, 4),
            "mask_seconds": round(mask_seconds, 4),
            "chunk_seconds": round(chunk_seconds, 4),
            "map_seconds": round(map_seconds, 4),
            "reduce_seconds": round(reduce_seconds, 4),
            "unmask_seconds": round(unmask_seconds, 4),
            "total_seconds": round(total_seconds, 4),
            "chunk_count": len(chunks),
            "batch_count": len(batch_groups),
        },
    }


# Optional compatibility alias if your app expects the old name.
async def generateResponse(pdf):
    return await generate_response(pdf)
