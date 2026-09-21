#!/usr/bin/env python3
"""Create a score-free cryptographic and tokenizer audit for an around-7B model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
from pathlib import Path
from typing import Any

from safetensors import safe_open
from transformers import AutoTokenizer

from experiments.run_natural_repeat import git_value, sha256_file
from experiments.run_babilong_prompt import (
    render_chat_prompt,
    validate_loaded_model,
)


ROOT = Path(__file__).resolve().parents[1]
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
)


def frozen_roster_entry(design: dict[str, Any], model_name: str) -> dict[str, Any]:
    matches = [row for row in design["frozen_model_roster"] if row["name"] == model_name]
    if len(matches) != 1:
        raise RuntimeError(f"model roster entry count is {len(matches)} for {model_name}")
    return matches[0]


def repository_environment() -> dict[str, Any]:
    return {
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "git_dirty": bool(git_value(["status", "--porcelain"])),
        "libraries": {
            name: importlib.metadata.version(name)
            for name in ("transformers", "tokenizers", "sentencepiece", "safetensors")
        },
    }


def validate_non_thinking_template(rendered: str) -> bool:
    """Accept Qwen3's official empty marker, but no active reasoning prefill."""
    blocks = re.findall(r"<think>(.*?)</think>", rendered, flags=re.DOTALL)
    if len(blocks) != 1 or blocks[0].strip():
        raise RuntimeError(
            "non-thinking template must contain exactly one empty closed think block"
        )
    if rendered.split("</think>", 1)[1].strip():
        raise RuntimeError("non-thinking template has non-whitespace prefill after </think>")
    return True


def frozen_weight_paths(model_dir: Path) -> tuple[list[Path], int | None, int | None]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        shards = sorted(set(index["weight_map"].values()))
        paths = [model_dir / shard for shard in shards]
        return paths, len(index["weight_map"]), int(index.get("metadata", {}).get("total_size", 0))
    single = model_dir / "model.safetensors"
    if not single.exists():
        raise RuntimeError("checkpoint has neither a safetensors index nor model.safetensors")
    return [single], None, None


def tensor_manifest(paths: list[Path]) -> dict[str, Any]:
    names: set[str] = set()
    parameters = 0
    payload_bytes = 0
    files: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"missing weight shard: {path}")
        shard_tensors = 0
        shard_parameters = 0
        shard_payload_bytes = 0
        with safe_open(path, framework="pt", device="cpu") as stream:
            for name in stream.keys():
                if name in names:
                    raise RuntimeError(f"tensor appears in more than one shard: {name}")
                names.add(name)
                tensor_slice = stream.get_slice(name)
                shape = tuple(int(value) for value in tensor_slice.get_shape())
                dtype = str(tensor_slice.get_dtype())
                if dtype not in DTYPE_BYTES:
                    raise RuntimeError(f"unsupported safetensors dtype: {dtype}")
                count = math.prod(shape)
                size = count * DTYPE_BYTES[dtype]
                parameters += count
                payload_bytes += size
                shard_tensors += 1
                shard_parameters += count
                shard_payload_bytes += size
        files.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "tensors": shard_tensors,
                "parameters": shard_parameters,
                "tensor_payload_bytes": shard_payload_bytes,
            }
        )
    return {
        "files": files,
        "tensors": len(names),
        "parameters": parameters,
        "tensor_payload_bytes": payload_bytes,
    }


def verify_official_weight_manifest(
    weights: dict[str, Any], roster: dict[str, Any]
) -> None:
    local_artifacts = [
        {"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]}
        for row in weights["files"]
    ]
    if local_artifacts != roster.get("official_weight_artifacts"):
        raise RuntimeError(
            "local checkpoint shards differ from the official pinned LFS manifest"
        )


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_official_critical_manifest(
    paths: list[Path], roster: dict[str, Any]
) -> list[dict[str, Any]]:
    expected_rows = roster.get("official_critical_artifacts", [])
    expected = {row["path"]: row for row in expected_rows}
    if len(expected) != len(expected_rows) or set(expected) != {path.name for path in paths}:
        raise RuntimeError("local critical-file set differs from the official manifest")
    verified: list[dict[str, Any]] = []
    for path in sorted(paths):
        row = expected[path.name]
        actual = {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "git_blob_sha1": git_blob_sha1(path),
        }
        if actual["bytes"] != row["bytes"]:
            raise RuntimeError(f"official critical-file byte mismatch: {path.name}")
        if "sha256" in row and actual["sha256"] != row["sha256"]:
            raise RuntimeError(f"official critical-file SHA-256 mismatch: {path.name}")
        if "git_blob_sha1" in row and actual["git_blob_sha1"] != row["git_blob_sha1"]:
            raise RuntimeError(f"official critical-file Git blob mismatch: {path.name}")
        if ("sha256" in row) == ("git_blob_sha1" in row):
            raise RuntimeError(f"official critical-file manifest digest is ambiguous: {path.name}")
        verified.append(actual)
    return verified


def validate_safetensors_index_metadata(
    weights: dict[str, Any],
    indexed_tensors: int | None,
    indexed_payload_bytes: int | None,
) -> dict[str, Any]:
    if indexed_tensors is None:
        return {
            "indexed_checkpoint": False,
            "tensor_count_match": True,
            "payload_bytes_match": True,
            "payload_delta_bytes": 0,
            "resolved_by_official_index_and_exact_model_load": True,
        }
    if weights["tensors"] != indexed_tensors:
        raise RuntimeError("safetensors index and shards disagree on tensor count")
    indexed_payload = int(indexed_payload_bytes or 0)
    actual_payload = int(weights["tensor_payload_bytes"])
    return {
        "indexed_checkpoint": True,
        "indexed_tensors": indexed_tensors,
        "actual_tensors": weights["tensors"],
        "tensor_count_match": True,
        "indexed_payload_bytes": indexed_payload,
        "actual_tensor_payload_bytes": actual_payload,
        "payload_bytes_match": actual_payload == indexed_payload,
        "payload_delta_bytes": actual_payload - indexed_payload,
        "resolved_by_official_index_and_exact_model_load": True,
    }


def validate_cuda_model_load(
    model_dir: Path,
    model_config: dict[str, Any],
    weights: dict[str, Any],
    *,
    registered_context: int,
) -> dict[str, Any]:
    """Exercise the exact BF16/CUDA model-load path without decoding a score."""
    import torch
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the score-free model-load audit")
    architectures = model_config.get("architectures", [])
    if len(architectures) != 1:
        raise RuntimeError("model config must register exactly one architecture")
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, dtype=torch.bfloat16
        )
        .to("cuda")
        .eval()
    )
    try:
        endpoint = {
            "model_parameters": weights["parameters"],
            "model_type": model_config.get("model_type"),
            "architecture": architectures[0],
        }
        identity = validate_loaded_model(
            model, endpoint, context_tokens=registered_context
        )
        if type(model).__name__ != architectures[0]:
            raise RuntimeError(
                f"loaded class {type(model).__name__!r} differs from registered "
                f"architecture {architectures[0]!r}"
            )
        parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
        if parameter_dtypes != ["torch.bfloat16"]:
            raise RuntimeError(f"unexpected loaded parameter dtypes: {parameter_dtypes}")
        return {
            **identity,
            "python_class": type(model).__name__,
            "parameter_dtypes": parameter_dtypes,
            "cuda_device": torch.cuda.get_device_name(),
            "generation_invoked": False,
            "load_pass": True,
        }
    finally:
        del model
        torch.cuda.empty_cache()


def audit(design_path: Path, model_name: str, model_dir: Path) -> dict[str, Any]:
    design = json.loads(design_path.read_text())
    if design["execution_authorization"]["authorized"] is not False:
        raise RuntimeError("score-free design must still be non-executable")
    roster = frozen_roster_entry(design, model_name)
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise RuntimeError("config.json is missing")
    model_config = json.loads(config_path.read_text())
    native_context = int(model_config.get("max_position_embeddings", 0))
    registered_context = int(design["source_panel_protocol"]["context_tokens"])
    if native_context < registered_context:
        raise RuntimeError(
            f"native context {native_context} is below registered {registered_context}"
        )

    weight_paths, indexed_tensors, indexed_payload_bytes = frozen_weight_paths(model_dir)
    weights = tensor_manifest(weight_paths)
    index_validation = validate_safetensors_index_metadata(
        weights, indexed_tensors, indexed_payload_bytes
    )
    verify_official_weight_manifest(weights, roster)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    template_kwargs = roster.get("chat_template_kwargs", {})
    probe_content = "Reply with exactly one lowercase location word: kitchen"
    rendered = render_chat_prompt(tokenizer, probe_content, roster)
    if rendered.count(probe_content) != 1:
        raise RuntimeError("chat template did not preserve the user probe exactly once")
    non_thinking_template_validated = False
    if template_kwargs.get("enable_thinking") is False:
        non_thinking_template_validated = validate_non_thinking_template(rendered)
    tokenized_probe = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    if not tokenized_probe:
        raise RuntimeError("chat-template probe tokenized to an empty sequence")
    model_load_validation = validate_cuda_model_load(
        model_dir,
        model_config,
        weights,
        registered_context=registered_context,
    )

    critical_files = [config_path]
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        critical_files.append(index_path)
    generation_config = model_dir / "generation_config.json"
    if generation_config.exists():
        critical_files.append(generation_config)
    critical_files.extend(model_dir / name for name in TOKENIZER_FILES if (model_dir / name).exists())
    critical_file_audit = verify_official_critical_manifest(
        sorted(set(critical_files)), roster
    )
    return {
        "schema_version": 1,
        "status": "score_free_checkpoint_and_template_audit",
        "model": roster,
        "design": {"path": str(design_path), "sha256": sha256_file(design_path)},
        "model_dir": str(model_dir),
        "model_config": {
            "path": "config.json",
            "sha256": sha256_file(config_path),
            "model_type": model_config.get("model_type"),
            "architectures": model_config.get("architectures", []),
            "native_context_tokens": native_context,
            "torch_dtype": model_config.get("torch_dtype"),
            "tie_word_embeddings": model_config.get("tie_word_embeddings"),
        },
        "weights": weights,
        "safetensors_index_validation": index_validation,
        "official_lfs_manifest_exact_match": True,
        "critical_files": critical_file_audit,
        "official_critical_manifest_exact_match": True,
        "chat_template": {
            "kwargs": template_kwargs,
            "frozen_system_prompt": roster.get("system_prompt"),
            "rendered_probe_sha256": __import__("hashlib").sha256(rendered.encode()).hexdigest(),
            "probe_tokens": len(tokenized_probe),
            "user_content_exactly_once": True,
            "thinking_disabled_when_registered": template_kwargs.get("enable_thinking") is False,
            "non_thinking_template_validated": non_thinking_template_validated,
        },
        "model_load_validation": model_load_validation,
        "environment": repository_environment(),
        "decoder_scores_observed": False,
        "audit_pass": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.design, args.model_name, args.model_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "model": result["model"]["name"],
        "model_config": result["model_config"],
        "weights": result["weights"],
        "chat_template": result["chat_template"],
        "audit_pass": result["audit_pass"],
    }, indent=2))


if __name__ == "__main__":
    main()
