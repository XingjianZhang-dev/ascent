#!/usr/bin/env python3
"""Evaluate Foundation and scale-expanded ASCENT memory on BABILong."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ascent.babilong_memory import (
    canonical_babilong_event_sources,
    extract_babilong_events,
    irrelevant_babilong_facts,
    project_babilong_inventory,
    read_babilong,
    resolve_babilong_event_sources,
)
from ascent.babilong_controls import matched_raw_event_fifo
from ascent.target_blindness import TargetBlindRow, fact_provenance_record
from ascent.generic_retrieval import retrieve_lexical_chain, split_passages
from experiments.run_natural_repeat import git_value, sha256_file
from experiments.run_ruler_niah import verify_model_artifact


LOCATIONS = ("bathroom", "kitchen", "bedroom", "garden", "hallway", "office")
LOCATION_PATTERN = re.compile(r"\b(" + "|".join(LOCATIONS) + r")\b", re.I)
COUNT_WORDS = ("none", "one", "two", "three")
COUNT_PATTERN = re.compile(r"\b(" + "|".join(COUNT_WORDS) + r")\b", re.I)
OBJECTS = ("milk", "apple", "football")
OBJECT_PATTERN = re.compile(r"\b(" + "|".join(OBJECTS) + r")\b", re.I)


def frozen_model_artifact_spec(
    endpoint: dict[str, Any],
) -> dict[str, Any] | list[dict[str, Any]]:
    """Return the one and only frozen weight-manifest field for an endpoint."""
    singular = endpoint.get("model_artifact")
    plural = endpoint.get("model_artifacts")
    if (singular is None) == (plural is None):
        raise RuntimeError(
            "endpoint must freeze exactly one of model_artifact or model_artifacts"
        )
    return singular if singular is not None else plural


def extract_location(text: str) -> str | None:
    match = LOCATION_PATTERN.search(text)
    return match.group(1).lower() if match else None


def extract_task_answer(text: str, task: str) -> str | None:
    """Extract the frozen task answer without consulting the reference target."""
    if task in {"qa1", "qa2", "qa3"}:
        return extract_location(text)
    if task == "qa7":
        match = COUNT_PATTERN.search(text)
        if match:
            return match.group(1).lower()
        digit_match = re.search(r"\b([0-3])\b", text)
        return COUNT_WORDS[int(digit_match.group(1))] if digit_match else None
    if task == "qa8":
        lowered = text.lower()
        if re.search(r"\bnothing\b", lowered):
            return "nothing"
        objects: list[str] = []
        for match in OBJECT_PATTERN.finditer(lowered):
            object_name = match.group(1).lower()
            if object_name not in objects:
                objects.append(object_name)
        return ",".join(objects) if objects else None
    raise ValueError(f"unsupported BABILong task: {task}")


def answer_instruction(task: str, *, neutral: bool = False) -> str:
    if task in {"qa1", "qa2", "qa3"}:
        return "Reply with exactly one lowercase location word."
    if task == "qa7":
        if neutral:
            return "Reply with the current object count as one lowercase word."
        return "Reply with exactly one lowercase number word: none, one, two, or three."
    if task == "qa8":
        if neutral:
            return (
                "Reply with only the lowercase object names in first-acquisition "
                "order, separated by commas. If no object remains, reply nothing."
            )
        return (
            "Reply with exactly 'nothing' or the lowercase object names separated "
            "by commas in chronological acquisition order (for example: "
            "apple,football)."
        )
    raise ValueError(f"unsupported BABILong task: {task}")


def corrupt_locations(text: str) -> str:
    mapping = dict(zip(LOCATIONS, LOCATIONS[1:] + LOCATIONS[:1], strict=True))
    return LOCATION_PATTERN.sub(lambda match: mapping[match.group(1).lower()], text)


def summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    standard_error = float(array.std(ddof=1) / np.sqrt(array.size))
    return {
        "samples": int(array.size),
        "mean": float(array.mean()),
        "standard_error": standard_error,
        "ci95_low": float(array.mean() - 1.96 * standard_error),
        "ci95_high": float(array.mean() + 1.96 * standard_error),
    }


def remaining_error_elimination(
    foundation_scores: list[float], ascent_scores: list[float]
) -> float | None:
    foundation_mean = float(np.mean(foundation_scores))
    ascent_mean = float(np.mean(ascent_scores))
    remaining_error = 1.0 - foundation_mean
    if remaining_error <= 0:
        return None
    return (ascent_mean - foundation_mean) / remaining_error


def prompt_input_token_budget(context_tokens: int, max_new_tokens: int) -> int:
    budget = context_tokens - max_new_tokens
    if budget <= 0:
        raise ValueError("max_new_tokens must be smaller than context_tokens")
    return budget


def configure_decoder_tokenizer(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"


def validate_loaded_model(
    model: Any, endpoint: dict[str, Any], *, context_tokens: int
) -> dict[str, Any]:
    """Fail closed when a loaded checkpoint differs from its frozen endpoint spec."""
    config = model.config
    model_type = str(getattr(config, "model_type", ""))
    architectures = tuple(getattr(config, "architectures", ()) or ())
    expected_model_type = endpoint.get("model_type")
    if expected_model_type is not None and model_type != expected_model_type:
        raise RuntimeError(
            f"model_type mismatch: loaded {model_type!r}, expected {expected_model_type!r}"
        )
    expected_architecture = endpoint.get("architecture")
    if expected_architecture is not None and expected_architecture not in architectures:
        raise RuntimeError(
            f"architecture mismatch: loaded {architectures!r}, expected "
            f"{expected_architecture!r}"
        )
    native_context = int(getattr(config, "max_position_embeddings", 0))
    if native_context < context_tokens:
        raise RuntimeError(
            f"native context {native_context} is smaller than registered "
            f"context {context_tokens}"
        )
    loaded_parameters = sum(parameter.numel() for parameter in model.parameters())
    expected_parameters = int(endpoint["model_parameters"])
    if loaded_parameters != expected_parameters:
        raise RuntimeError(
            f"parameter-count mismatch: loaded {loaded_parameters}, expected "
            f"{expected_parameters}"
        )
    return {
        "model_type": model_type,
        "architectures": list(architectures),
        "native_context_tokens": native_context,
        "loaded_parameters": loaded_parameters,
    }


def validate_runtime_dependencies(config: dict[str, Any]) -> dict[str, str]:
    expected = config.get("runtime_dependencies")
    if expected is None:
        return {}
    actual = {name: importlib.metadata.version(name) for name in expected}
    if actual != expected:
        raise RuntimeError(
            f"runtime dependency mismatch: loaded {actual!r}, expected {expected!r}"
        )
    return actual


def render_chat_prompt(tokenizer: Any, prompt: str, endpoint: dict[str, Any]) -> str:
    """Apply only endpoint-frozen chat-template switches."""
    kwargs = endpoint.get("chat_template_kwargs", {})
    if not isinstance(kwargs, dict) or any(not isinstance(key, str) for key in kwargs):
        raise RuntimeError("chat_template_kwargs must be a string-keyed mapping")
    system_prompt = endpoint.get("system_prompt")
    if system_prompt is not None and (
        not isinstance(system_prompt, str) or not system_prompt.strip()
    ):
        raise RuntimeError("system_prompt must be a non-empty string when registered")
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )


def certified_state_answer(
    task: str, structured_count: str | None, structured_inventory: str | None
) -> str:
    if task == "qa7" and structured_count is not None:
        return structured_count
    if task == "qa8" and structured_inventory is not None:
        return structured_inventory
    raise ValueError("certified state answer requires QA7/QA8 structured state")


def optional_certified_state_answer(
    readout_path: str,
    task: str,
    structured_count: str | None,
    structured_inventory: str | None,
) -> str | None:
    """Construct certified output only for the registered certified readout."""
    if readout_path != "certified_evidence":
        return None
    return certified_state_answer(task, structured_count, structured_inventory)


def run(
    config_path: Path,
    data_path: Path,
    endpoint_name: str,
    model_dir: Path,
    output: Path,
    condition_name: str | None = None,
    decode_order: str = "foundation_first",
    diagnostic_batch_size: int | None = None,
    audit_rerun: bool = False,
    audit_label: str = "posthoc_code_audit_correction_rerun_no_independence_claim",
) -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_wall_start = time.perf_counter()
    config: dict[str, Any] = json.loads(config_path.read_text())
    validate_runtime_dependencies(config)
    endpoint = {row["name"]: row for row in config["endpoints"]}[endpoint_name]
    condition: dict[str, Any] = {}
    if "conditions" in config:
        if condition_name is None or condition_name not in config["conditions"]:
            raise RuntimeError("a frozen robustness --condition is required")
        condition = config["conditions"][condition_name]
    allowed_decode_orders = config.get("allowed_decode_orders", ["foundation_first"])
    if decode_order not in allowed_decode_orders and diagnostic_batch_size is None:
        raise RuntimeError("decode order is absent from the frozen config")
    expected_panel_hash = config.get("panel_sha256")
    if "panel_sha256_by_name" in config:
        expected_panel_hash = config["panel_sha256_by_name"].get(data_path.stem)
        if expected_panel_hash is None:
            raise RuntimeError("BABILong panel is absent from the frozen hash map")
    if sha256_file(data_path) != expected_panel_hash:
        raise RuntimeError("BABILong panel hash mismatch")
    rows = [json.loads(line) for line in data_path.read_text().splitlines()]
    if len(rows) != int(config["evaluation_samples"]):
        raise RuntimeError("unexpected BABILong panel row count")
    tasks = tuple(config.get("tasks", ("qa1", "qa2", "qa3")))
    if not tasks or any(row["task"] not in tasks for row in rows):
        raise RuntimeError("panel contains a task absent from the frozen config")
    task_counts = {task: sum(row["task"] == task for row in rows) for task in tasks}
    if len(set(task_counts.values())) != 1:
        raise RuntimeError(f"panel is not task-balanced: {task_counts}")

    artifact_start = time.perf_counter()
    model_files = verify_model_artifact(
        model_dir, frozen_model_artifact_spec(endpoint)
    )
    artifact_verification_seconds = time.perf_counter() - artifact_start
    model_load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    configure_decoder_tokenizer(tokenizer)
    context_tokens = int(config["context_tokens"])
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, dtype=torch.bfloat16
        )
        .to("cuda")
        .eval()
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model_identity = validate_loaded_model(
        model, endpoint, context_tokens=context_tokens
    )
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - model_load_start
    torch.manual_seed(int(config["seed"]))
    torch.cuda.reset_peak_memory_stats()

    fact_slots = int(
        condition.get("state_fact_slots_by_endpoint", {}).get(
            endpoint_name, endpoint["state_fact_slots"]
        )
    )
    memory_source = str(condition.get("memory_source", "relevant"))
    memory_representation = str(condition.get("memory_representation", "event_facts"))
    if memory_representation not in {
        "canonical_event_facts",
        "event_facts",
        "resolved_event_facts",
        "structured_inventory",
    }:
        raise RuntimeError(
            f"unsupported memory representation: {memory_representation}"
        )
    neutral_answer_format = bool(condition.get("neutral_answer_format", False))
    readout_path = str(condition.get("readout_path", "foundation_generation"))
    if readout_path not in {"foundation_generation", "certified_evidence"}:
        raise RuntimeError(f"unsupported readout path: {readout_path}")
    if (
        readout_path == "certified_evidence"
        and memory_representation != "structured_inventory"
    ):
        raise RuntimeError("certified evidence requires structured inventory state")
    if (
        memory_representation in {"canonical_event_facts", "resolved_event_facts"}
        and memory_source != "relevant"
    ):
        raise RuntimeError("canonical/resolved event facts require relevant causal memory")
    if memory_source not in {
        "relevant",
        "irrelevant",
        "corrupted",
        "generic_bm25",
        "matched_raw_event_fifo",
        "bge_m3_hybrid_rerank",
        "sentence_window",
    }:
        raise RuntimeError(f"unsupported memory source: {memory_source}")
    retrieval_cache: dict[str, dict[str, Any]] = {}
    retrieval_cache_metadata: dict[str, Any] | None = None
    if memory_source == "bge_m3_hybrid_rerank":
        cache_spec = condition.get("retrieval_cache_by_panel", {}).get(data_path.stem)
        if cache_spec is None:
            raise RuntimeError("frozen BGE-M3 retrieval cache is missing")
        cache_path = Path(cache_spec["path"])
        if not cache_path.is_absolute():
            cache_path = ROOT / cache_path
        cache_hash = sha256_file(cache_path)
        if cache_hash != cache_spec["sha256"]:
            raise RuntimeError("BGE-M3 retrieval cache hash mismatch")
        cache_document = json.loads(cache_path.read_text())
        if cache_document["data_sha256"] != sha256_file(data_path):
            raise RuntimeError("BGE-M3 retrieval cache data mismatch")
        retrieval_cache = {row["row_id"]: row for row in cache_document["rows"]}
        retrieval_cache_metadata = {
            "path": str(cache_path),
            "sha256": cache_hash,
            "protocol_sha256": cache_document["protocol_sha256"],
            "embedding_model_revision": cache_document["embedding_model"]["revision"],
            "reranker_model_revision": cache_document["reranker_model"]["revision"],
        }
    max_new_tokens = int(config["max_new_tokens"])
    input_token_budget = prompt_input_token_budget(context_tokens, max_new_tokens)
    frozen_batch_size = int(endpoint["batch_size"])
    batch_size = frozen_batch_size
    diagnostic_status = None
    if audit_rerun:
        # Legacy default label kept for traceability of retained records; new
        # audit reruns pass an explicit --audit-label describing the audit.
        diagnostic_status = audit_label
    if diagnostic_batch_size is not None:
        if diagnostic_batch_size <= 0:
            raise RuntimeError("diagnostic batch size must be positive")
        batch_size = diagnostic_batch_size
        diagnostic_status = "posthoc_decode_batch_diagnostic_no_promotion"
    prepared: list[dict[str, Any]] = []
    parser_failures: list[str] = []
    retrieval_start = time.perf_counter()
    for raw_row in rows:
        # Target-blindness instrumentation (C-08): the writer, readout, and prompt
        # builder only ever see this view, on which reading ``target`` raises until
        # ``reveal()`` is called after the ASCENT prompt has been rendered.
        row = TargetBlindRow(raw_row)
        read = read_babilong(
            row["input"],
            row["question"],
            history_slots=int(config["history_slots"]),
            enabled_tasks=tasks,
        )
        retained = read.facts[-fact_slots:]
        generic_retrieval = None
        matched_raw = None
        cached_neural = None
        if memory_source == "irrelevant":
            retained = irrelevant_babilong_facts(row["input"], read, count=fact_slots)
        if memory_source == "generic_bm25":
            generic_retrieval = retrieve_lexical_chain(
                row["input"], row["question"], count=fact_slots
            )
        sentence_window = None
        if memory_source == "sentence_window":
            # Schema-blind recency writer: the final ``fact_slots`` generic
            # punctuation-boundary sentences of the text, in document order. It
            # reads neither the question nor any task vocabulary.
            sentence_window = [
                sentence for _, sentence in split_passages(row["input"])[-fact_slots:]
            ]
        if memory_source == "matched_raw_event_fifo":
            matched_raw = matched_raw_event_fifo(
                row["input"],
                row["question"],
                count=fact_slots,
                budget_bytes=read.persistent_payload_bytes,
            )
        if memory_source == "bge_m3_hybrid_rerank":
            cached_neural = retrieval_cache.get(row["row_id"])
            if cached_neural is None:
                raise RuntimeError(f"BGE-M3 cache row missing: {row['row_id']}")
        output_instruction = answer_instruction(
            row["task"], neutral=neutral_answer_format
        )
        foundation_prompt = (
            "Find the short facts hidden in the distractor text and answer the question. "
            f"{output_instruction}\n\n"
            f"Text:\n{row['input']}\n\nQuestion: {row['question']}\nAnswer:"
        )
        retained_sources = [fact.source for fact in retained]
        if memory_representation == "resolved_event_facts":
            resolved_sources = resolve_babilong_event_sources(row["input"])
            retained_sources = [
                resolved_sources[fact.character_position] for fact in retained
            ]
        if memory_representation == "canonical_event_facts":
            canonical_sources = canonical_babilong_event_sources(row["input"])
            retained_sources = [
                canonical_sources[fact.character_position] for fact in retained
            ]
        if generic_retrieval is not None:
            retained_sources = list(generic_retrieval.passages)
        if sentence_window is not None:
            retained_sources = list(sentence_window)
        if matched_raw is not None:
            retained_sources = list(matched_raw.passages)
        if cached_neural is not None:
            retained_sources = [
                item["passage"]
                for item in sorted(
                    cached_neural["ranked_passages"][:fact_slots],
                    key=lambda item: item["character_position"],
                )
            ]
        if memory_source == "irrelevant":
            retained_sources.extend(
                ["No query-relevant fact was stored."]
                * (fact_slots - len(retained_sources))
            )
        if memory_source == "corrupted":
            retained_sources = [
                corrupt_locations(source) for source in retained_sources
            ]
        structured_inventory = None
        structured_count = None
        if memory_representation == "structured_inventory":
            projection_facts = retained
            if (
                generic_retrieval is not None
                or sentence_window is not None
                or matched_raw is not None
                or cached_neural is not None
            ):
                projection_facts = extract_babilong_events("\n".join(retained_sources))
            projection = project_babilong_inventory(
                tuple(projection_facts), row["question"]
            )
            structured_inventory = projection.inventory_text
            structured_count = projection.count_word
            event_text = "\n".join(f"- {source}" for source in retained_sources)
            memory_text = (
                f"Retained events:\n{event_text}\n"
                f"Reconstructed inventory: {structured_inventory}\n"
                f"Reconstructed count: {structured_count}"
            )
        else:
            memory_text = "\n".join(f"- {source}" for source in retained_sources)
        if generic_retrieval is not None:
            memory_instruction = (
                "A generic lexical retriever selected the following passages in "
                "chronological order. Answer using only these passages."
            )
        elif sentence_window is not None:
            memory_instruction = (
                "A task-agnostic recency window retained the final sentences of "
                "the text in chronological order. Answer using only these sentences."
            )
        elif matched_raw is not None:
            memory_instruction = (
                "A byte-matched raw event FIFO selected the following cached "
                "passages in chronological order. Answer using only these passages."
            )
        elif cached_neural is not None:
            memory_instruction = (
                "BGE-M3 hybrid retrieval and its official cross-encoder reranker "
                "selected the following passages in chronological order. Answer "
                "using only these passages."
            )
        elif memory_representation == "structured_inventory":
            memory_instruction = (
                "A target-blind causal memory replayed the retained event suffix "
                "into the following reconstructed state. Use the reconstructed "
                "field that matches the question."
            )
        elif memory_representation == "resolved_event_facts":
            memory_instruction = (
                "A target-blind causal memory resolved stream-local references "
                "at write time and retained the following explicit events in "
                "chronological order. Answer using only these events."
            )
        elif memory_representation == "canonical_event_facts":
            memory_instruction = (
                "A target-blind causal memory canonicalized stream events and "
                "retained them in chronological order without adding inferred "
                "location fields. Answer using only these events."
            )
        else:
            memory_instruction = (
                "A bounded causal memory retained the following relevant facts in "
                "chronological order. Answer using only these facts."
            )
        ascent_prompt = (
            f"{memory_instruction} {output_instruction}\n\n"
            f"Memory:\n{memory_text}\n\nQuestion: {row['question']}\nAnswer:"
        )
        provenance = (
            fact_provenance_record(row["input"], tuple(retained), list(retained_sources))
            if memory_source == "relevant"
            and memory_representation in {"event_facts", "canonical_event_facts"}
            else {"not_applicable": f"{memory_source}/{memory_representation}"}
        )
        row.reveal("parser_audit_after_state_and_prompt_construction")
        if read.answer != row["target"]:
            parser_failures.append(row["row_id"])
        target_blindness = {**row.record(), "state_provenance": provenance}
        if target_blindness["target_read_before_reveal"] or target_blindness["blocked_target_access_attempts"]:
            raise RuntimeError(
                f"target-blindness violation on row {row['row_id']}: {target_blindness}"
            )
        prepared.append(
            {
                **raw_row,
                "target_blindness": target_blindness,
                "foundation_prompt": foundation_prompt,
                "ascent_prompt": ascent_prompt,
                "parser_answer": read.answer,
                "supporting_fact_count": read.supporting_fact_count,
                "retained_fact_count": len(retained_sources),
                "retained_facts": retained_sources,
                "structured_inventory": structured_inventory,
                "structured_count": structured_count,
                "certified_answer": optional_certified_state_answer(
                    readout_path,
                    row["task"],
                    structured_count,
                    structured_inventory,
                ),
                "retained_read_state_bytes": (
                    len(memory_text.encode("utf-8"))
                    if memory_representation == "structured_inventory"
                    else sum(
                        len(source.encode("utf-8")) + 24 for source in retained_sources
                    )
                ),
                "persistent_payload_bytes": (
                    generic_retrieval.corpus_utf8_bytes
                    if generic_retrieval is not None
                    else sum(len(source.encode("utf-8")) for source in retained_sources)
                    if sentence_window is not None
                    else (
                        matched_raw.payload_bytes
                        if matched_raw is not None
                        else (
                            cached_neural["index_payload_bytes_lower_bound"]
                            if cached_neural is not None
                            else read.persistent_payload_bytes
                        )
                    )
                ),
                "write_budget_bytes": (
                    matched_raw.budget_bytes if matched_raw is not None else None
                ),
                "write_payload_utilization": (
                    matched_raw.payload_bytes / matched_raw.budget_bytes
                    if matched_raw is not None
                    else None
                ),
            }
        )
    retrieval_seconds = time.perf_counter() - retrieval_start
    if parser_failures:
        raise RuntimeError(f"causal parser mismatch on {len(parser_failures)} rows")

    def decode(
        prompt_key: str, *, limit: int | None = None
    ) -> tuple[list[str], list[int], dict[str, Any]]:
        decoded: list[str] = []
        token_lengths: list[int] = []
        generated_token_count = 0
        decode_rows = prepared if limit is None else prepared[:limit]
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        allocated_before = int(torch.cuda.memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        decode_start = time.perf_counter()
        for start in range(0, len(decode_rows), batch_size):
            prompts = [
                row[prompt_key] for row in decode_rows[start : start + batch_size]
            ]
            chat_prompts = [render_chat_prompt(tokenizer, prompt, endpoint) for prompt in prompts]
            encoded = tokenizer(
                chat_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=input_token_budget,
                add_special_tokens=False,
            ).to("cuda")
            token_lengths.extend(
                int(value) for value in encoded.attention_mask.sum(dim=1)
            )
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            suffix = generated[:, encoded.input_ids.shape[1] :]
            generated_token_count += int(
                (suffix != tokenizer.pad_token_id).sum().item()
            )
            decoded.extend(tokenizer.batch_decode(suffix, skip_special_tokens=True))
        torch.cuda.synchronize()
        seconds = time.perf_counter() - decode_start
        peak_bytes = int(torch.cuda.max_memory_allocated())
        prompt_token_count = sum(token_lengths)
        metrics = {
            "seconds": seconds,
            "samples": len(decode_rows),
            "samples_per_second": len(decode_rows) / seconds,
            "prompt_tokens": prompt_token_count,
            "prompt_token_cap": input_token_budget,
            "prompt_truncation_side": tokenizer.truncation_side,
            "maximum_prompt_tokens": max(token_lengths),
            "samples_at_prompt_token_cap": sum(
                length == input_token_budget for length in token_lengths
            ),
            "generated_tokens": generated_token_count,
            "prompt_tokens_per_second": prompt_token_count / seconds,
            "allocated_before_bytes": allocated_before,
            "peak_allocated_bytes": peak_bytes,
            "incremental_peak_bytes": peak_bytes - allocated_before,
        }
        return decoded, token_lengths, metrics

    started = datetime.now(timezone.utc).isoformat()
    wall_start = time.perf_counter()
    warmup_samples = int(config.get("systems_warmup_samples", 0))
    if warmup_samples:
        if not 0 < warmup_samples < len(prepared):
            raise RuntimeError("invalid frozen systems warmup size")
        decode("foundation_prompt", limit=warmup_samples)
        if readout_path == "foundation_generation":
            decode("ascent_prompt", limit=warmup_samples)
    if readout_path == "certified_evidence":
        foundation_outputs, foundation_token_lengths, foundation_systems = decode(
            "foundation_prompt"
        )
        certified_start = time.perf_counter()
        ascent_outputs = [row["certified_answer"] for row in prepared]
        certified_seconds = time.perf_counter() - certified_start
        ascent_token_lengths = [0] * len(prepared)
        ascent_systems = {
            "seconds": certified_seconds,
            "samples": len(prepared),
            "samples_per_second": len(prepared) / max(certified_seconds, 1e-12),
            "prompt_tokens": 0,
            "prompt_token_cap": 0,
            "prompt_truncation_side": "not_applicable",
            "maximum_prompt_tokens": 0,
            "samples_at_prompt_token_cap": 0,
            "generated_tokens": 0,
            "prompt_tokens_per_second": 0.0,
            "allocated_before_bytes": 0,
            "peak_allocated_bytes": 0,
            "incremental_peak_bytes": 0,
        }
    elif decode_order == "foundation_first":
        foundation_outputs, foundation_token_lengths, foundation_systems = decode(
            "foundation_prompt"
        )
        ascent_outputs, ascent_token_lengths, ascent_systems = decode("ascent_prompt")
    else:
        ascent_outputs, ascent_token_lengths, ascent_systems = decode("ascent_prompt")
        foundation_outputs, foundation_token_lengths, foundation_systems = decode(
            "foundation_prompt"
        )

    foundation_scores: list[float] = []
    ascent_scores: list[float] = []
    predictions: list[dict[str, Any]] = []
    by_task: dict[str, dict[str, list[float]]] = {
        task: {"foundation": [], "ascent": []} for task in tasks
    }
    for index, row in enumerate(prepared):
        foundation_answer = extract_task_answer(foundation_outputs[index], row["task"])
        ascent_answer = extract_task_answer(ascent_outputs[index], row["task"])
        foundation_score = float(foundation_answer == row["target"])
        ascent_score = float(ascent_answer == row["target"])
        foundation_scores.append(foundation_score)
        ascent_scores.append(ascent_score)
        by_task[row["task"]]["foundation"].append(foundation_score)
        by_task[row["task"]]["ascent"].append(ascent_score)
        predictions.append(
            {
                "row_id": row["row_id"],
                "task": row["task"],
                "target": row["target"],
                "foundation_output": foundation_outputs[index],
                "foundation_answer": foundation_answer,
                "foundation_location": (
                    foundation_answer if row["task"] in {"qa1", "qa2", "qa3"} else None
                ),
                "foundation_score": foundation_score,
                "ascent_output": ascent_outputs[index],
                "ascent_answer": ascent_answer,
                "ascent_location": (
                    ascent_answer if row["task"] in {"qa1", "qa2", "qa3"} else None
                ),
                "ascent_score": ascent_score,
                "supporting_fact_count": row["supporting_fact_count"],
                "retained_fact_count": row["retained_fact_count"],
                "retained_facts": row["retained_facts"],
                "structured_inventory": row["structured_inventory"],
                "structured_count": row["structured_count"],
                "retained_read_state_bytes": row["retained_read_state_bytes"],
                "persistent_payload_bytes": row["persistent_payload_bytes"],
                "write_budget_bytes": row["write_budget_bytes"],
                "write_payload_utilization": row["write_payload_utilization"],
                "foundation_prompt_tokens": foundation_token_lengths[index],
                "ascent_prompt_tokens": ascent_token_lengths[index],
                "target_blindness": row["target_blindness"],
            }
        )

    gains = [a - f for a, f in zip(ascent_scores, foundation_scores, strict=True)]
    result = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "status": diagnostic_status or config["status"],
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - wall_start,
        "end_to_end_seconds": time.perf_counter() - run_wall_start,
        "endpoint": endpoint,
        "condition": {
            "name": condition_name or "primary_relevant",
            "memory_source": memory_source,
            "memory_representation": memory_representation,
            "neutral_answer_format": neutral_answer_format,
            "readout_path": readout_path,
            "fact_slots": fact_slots,
            "frozen_batch_size": frozen_batch_size,
            "effective_batch_size": batch_size,
            "diagnostic_no_promotion": diagnostic_status is not None,
        },
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "data": {
            "path": str(data_path),
            "sha256": sha256_file(data_path),
            "rows": len(rows),
            "task_counts": task_counts,
        },
        "model_files": model_files,
        "model_identity": model_identity,
        "parser_accuracy": 1.0,
        "foundation": summary(foundation_scores),
        "ascent": summary(ascent_scores),
        "gain": summary(gains),
        "remaining_error_elimination": remaining_error_elimination(
            foundation_scores, ascent_scores
        ),
        "wins": sum(
            a > f for a, f in zip(ascent_scores, foundation_scores, strict=True)
        ),
        "regressions": sum(
            a < f for a, f in zip(ascent_scores, foundation_scores, strict=True)
        ),
        "by_task": {
            task: {
                "foundation": summary(values["foundation"]),
                "ascent": summary(values["ascent"]),
                "gain": summary(
                    [
                        a - f
                        for a, f in zip(
                            values["ascent"], values["foundation"], strict=True
                        )
                    ]
                ),
                "remaining_error_elimination": remaining_error_elimination(
                    values["foundation"], values["ascent"]
                ),
            }
            for task, values in by_task.items()
        },
        "state": {
            "fact_slots": fact_slots,
            "write_state_payload_bytes": {
                "mean": float(
                    np.mean([row["persistent_payload_bytes"] for row in prepared])
                ),
                "maximum": int(
                    max(row["persistent_payload_bytes"] for row in prepared)
                ),
                "scale_policy": {
                    "generic_bm25": "full-corpus UTF-8 lower bound for generic lexical index",
                    "matched_raw_event_fifo": "exact UTF-8 raw FIFO payload under each sample's ASCENT write budget",
                    "bge_m3_hybrid_rerank": "full-corpus UTF-8 plus float16 dense-index lower bound",
                }.get(memory_source, "fixed causal world state across endpoints"),
            },
            "read_state_payload_bytes": {
                "mean": float(
                    np.mean([row["retained_read_state_bytes"] for row in prepared])
                ),
                "maximum": int(
                    max(row["retained_read_state_bytes"] for row in prepared)
                ),
                "scale_policy": "query-conditioned fact slots grow with endpoint",
            },
            "mean_total_external_state_bytes": float(
                np.mean([row["persistent_payload_bytes"] for row in prepared])
                + np.mean([row["retained_read_state_bytes"] for row in prepared])
            ),
            "state_slots_per_model_parameter": fact_slots
            / int(endpoint["model_parameters"]),
        },
        "systems": {
            "decode_order": decode_order,
            "warmup_samples_per_arm": warmup_samples,
            "artifact_verification_seconds": artifact_verification_seconds,
            "model_load_seconds": model_load_seconds,
            "retrieval_total_seconds": retrieval_seconds,
            "retrieval_microseconds_per_sample": retrieval_seconds
            * 1e6
            / len(prepared),
            "foundation_decode": foundation_systems,
            "ascent_decode": ascent_systems,
            "ascent_to_foundation_decode_time_ratio": ascent_systems["seconds"]
            / foundation_systems["seconds"],
            "ascent_to_foundation_prompt_token_ratio": ascent_systems["prompt_tokens"]
            / foundation_systems["prompt_tokens"],
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "tokenizers": importlib.metadata.version("tokenizers"),
            "sentencepiece": importlib.metadata.version("sentencepiece"),
            "safetensors": importlib.metadata.version("safetensors"),
            "cuda_device": torch.cuda.get_device_name(0),
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
            "git_commit": git_value(["rev-parse", "HEAD"]),
            "git_dirty": bool(git_value(["status", "--porcelain"])),
        },
        "retrieval_cache": retrieval_cache_metadata,
        "preflight": {
            "target_blindness": {
                "rows": len(prepared),
                "blinded_keys": ["target"],
                "rows_with_blocked_target_access_attempts": sum(
                    1 for item in prepared if item["target_blindness"]["blocked_target_access_attempts"]
                ),
                "rows_with_target_read_before_reveal": sum(
                    1 for item in prepared if item["target_blindness"]["target_read_before_reveal"]
                ),
                "keys_read_before_reveal_union": sorted(
                    {key for item in prepared for key in item["target_blindness"]["keys_read_before_reveal"]}
                ),
                "revealed_phase": "parser_audit_after_state_and_prompt_construction",
                "rows_with_state_derived_from_input_only": sum(
                    1
                    for item in prepared
                    if item["target_blindness"]["state_provenance"].get("state_derived_from_input_only") is True
                ),
                "rows_with_provenance_not_applicable": sum(
                    1 for item in prepared if "not_applicable" in item["target_blindness"]["state_provenance"]
                ),
            },
            "target_values_not_rendered_by_prompt_builder": all(
                not item["target_blindness"]["target_read_before_reveal"]
                and item["target_blindness"]["blocked_target_access_attempts"] == 0
                for item in prepared
            ),
        },
        "predictions": predictions,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "endpoint": endpoint_name,
                "foundation": result["foundation"]["mean"],
                "ascent": result["ascent"]["mean"],
                "gain": result["gain"]["mean"],
                "wins": result["wins"],
                "regressions": result["regressions"],
                "output": str(output),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition")
    parser.add_argument(
        "--decode-order",
        choices=("foundation_first", "ascent_first"),
        default="foundation_first",
    )
    parser.add_argument(
        "--diagnostic-batch-size",
        type=int,
        help="posthoc decode diagnostic only; result is ineligible for promotion",
    )
    parser.add_argument(
        "--audit-rerun",
        action="store_true",
        help="tag a posthoc audit rerun as ineligible for independence claims",
    )
    parser.add_argument(
        "--audit-label",
        default="posthoc_code_audit_correction_rerun_no_independence_claim",
        help="status label written to an --audit-rerun record (default keeps the legacy label)",
    )
    args = parser.parse_args()
    run(
        args.config,
        args.data,
        args.endpoint,
        args.model_dir,
        args.output,
        args.condition,
        args.decode_order,
        args.diagnostic_batch_size,
        args.audit_rerun,
        args.audit_label,
    )


if __name__ == "__main__":
    main()
