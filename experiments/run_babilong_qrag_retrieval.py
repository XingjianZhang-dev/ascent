#!/usr/bin/env python3
"""Run a released Q-RAG checkpoint on one frozen ASCENT BABILong panel.

The script adapts only the dataset interface. Agent construction, checkpoint
loading, greedy action selection, relative positions, six-step budget, and the
published Q-value stopping rule are inherited from the frozen Q-RAG release.
It intentionally does not calculate answer accuracy; reader evaluation is a
separate frozen stage.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import re
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from experiments.run_natural_repeat import git_value, sha256_file


ROOT = Path(__file__).resolve().parents[1]


class RetrievalOnlyFeedback:
    """Concrete no-op feedback matching Q-RAG's intended dummy semantics."""

    def reset(self, obs: Any, info: Any) -> None:
        del obs, info

    def get_feedback(
        self, obs: Any, info: Any, truncated: bool = False
    ) -> dict[str, float | bool]:
        del obs, info, truncated
        return {"reward": 0.0, "terminated": False}

    def copy(self) -> "RetrievalOnlyFeedback":
        return RetrievalOnlyFeedback()


def install_unused_vllm_import_shim() -> bool:
    """Allow retrieval-only use without installing Q-RAG's reader runtime.

    Q-RAG imports its optional vLLM reader at package-import time even though
    the retrieval path never constructs it. Installing vLLM would replace the
    frozen Torch/Transformers stack. The shim exposes only the two names needed
    for import and fails loudly if either is ever instantiated.
    """
    try:
        __import__("vllm")
        return False
    except ModuleNotFoundError:
        pass

    class _UnusedVllmComponent:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("retrieval-only vLLM shim was instantiated")

    shim = types.ModuleType("vllm")
    shim.LLM = _UnusedVllmComponent  # type: ignore[attr-defined]
    shim.SamplingParams = _UnusedVllmComponent  # type: ignore[attr-defined]
    sys.modules["vllm"] = shim
    return True


def sentence_chunks(text: str, pattern: str) -> list[str]:
    """Apply the prospectively frozen sentence interface."""
    chunks = [part.strip() for part in re.split(pattern, text) if part.strip()]
    if not chunks:
        raise RuntimeError("Q-RAG sentence chunking produced an empty document")
    return chunks


def qvalue_keep_count(q_values: list[float], threshold: float) -> int:
    """Mirror Q-RAG's QValueChunkFilter: stop at the first Q <= threshold."""
    count = 0
    for value in q_values:
        if value <= threshold:
            break
        count += 1
    return count


def frozen_panel_rows(
    manifest: dict[str, Any], data_path: Path, task: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_path = ROOT / manifest["source_panel_config"]["path"]
    if sha256_file(source_path) != manifest["source_panel_config"]["sha256"]:
        raise RuntimeError("frozen Qwen panel config hash mismatch")
    source = json.loads(source_path.read_text())
    if data_path.stem not in manifest["evaluation"]["panels"]:
        raise RuntimeError("panel is absent from the Q-RAG peer manifest")
    if sha256_file(data_path) != source["panel_sha256_by_name"][data_path.stem]:
        raise RuntimeError("BABILong panel hash mismatch")
    if task not in manifest["evaluation"]["tasks"]:
        raise RuntimeError("task is absent from the Q-RAG peer manifest")
    rows = [json.loads(line) for line in data_path.read_text().splitlines()]
    selected = [row for row in rows if row["task"] == task]
    if len(selected) != int(manifest["evaluation"]["rows_per_task_per_panel"]):
        raise RuntimeError("unexpected task row count")
    return selected, source


def verify_qrag_release(
    manifest: dict[str, Any], qrag_root: Path, checkpoint_dir: Path, task: str
) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(qrag_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != manifest["qrag_code"]["commit"]:
        raise RuntimeError("Q-RAG code commit mismatch")
    if subprocess.run(
        ["git", "-C", str(qrag_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip():
        raise RuntimeError("Q-RAG worktree is dirty")
    spec = manifest["qrag_checkpoints"][task]
    config_path = checkpoint_dir / "config.yaml"
    model_path = checkpoint_dir / spec["model_path"]
    if sha256_file(config_path) != spec["config_sha256"]:
        raise RuntimeError("Q-RAG checkpoint config hash mismatch")
    if model_path.stat().st_size != int(spec["model_bytes"]):
        raise RuntimeError("Q-RAG checkpoint byte count mismatch")
    if sha256_file(model_path) != spec["model_sha256"]:
        raise RuntimeError("Q-RAG checkpoint hash mismatch")
    return spec


def run(
    manifest_path: Path,
    data_path: Path,
    task: str,
    qrag_root: Path,
    checkpoint_dir: Path,
    output: Path,
) -> None:
    import torch
    import transformers
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    started = datetime.now(timezone.utc).isoformat()
    wall_start = time.perf_counter()
    manifest = json.loads(manifest_path.read_text())
    if not manifest["status"].startswith("prospective_before_any_qrag_score"):
        raise RuntimeError("Q-RAG peer manifest is not prospectively frozen")
    rows, _ = frozen_panel_rows(manifest, data_path, task)
    checkpoint_spec = verify_qrag_release(
        manifest, qrag_root, checkpoint_dir, task
    )

    vllm_import_shim = install_unused_vllm_import_shim()
    sys.path.insert(0, str(qrag_root))
    from envs.qa_env import QAEnv  # type: ignore
    from rl.agents.pqn import PQN  # type: ignore

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_device("cuda:0")
    torch.set_float32_matmul_precision("high")

    checkpoint_config_path = checkpoint_dir / "config.yaml"
    config = OmegaConf.load(checkpoint_config_path)
    base_embedder = manifest["qrag_base_embedder"]
    if str(config.algo.model.model_name) != base_embedder["repo_id"]:
        raise RuntimeError("Q-RAG base embedder repository mismatch")
    # The released YAML points at the mutable `main` branch. Resolve every
    # downstream interpolation only after replacing it with the prospectively
    # frozen immutable commit.
    config.algo.model.revision = base_embedder["revision"]
    OmegaConf.resolve(config)
    agent = PQN(config.algo)
    agent.load(checkpoint_dir / checkpoint_spec["model_path"], strict=True)
    agent.eval()

    retrieval = manifest["evaluation"]["retrieval"]
    chunking = manifest["evaluation"]["chunking"]
    environment = QAEnv(
        dataset=None,
        max_steps=int(retrieval["max_steps"]),
        positions_processor=instantiate(
            config.envs.positions_processor_dict.relative
        ),
        action_embed_length=int(config.max_action_length),
        max_action_length_in_memory=int(config.max_action_length_in_memory),
        feedback_model=RetrievalOnlyFeedback(),
        separator=str(config.envs.env.separator),
        sort_by_index=bool(retrieval["sort_selected_chunks_by_document_index"]),
    )

    outputs: list[dict[str, Any]] = []
    retrieval_start = time.perf_counter()
    for row in rows:
        chunks = sentence_chunks(row["input"], chunking["regex"])
        sample = {
            "id": row["row_id"],
            "question": row["question"],
            "answer": row["target"],
            "chunks": chunks,
            "sf_idx": [],
        }
        state = environment.reset(sample)
        embeds, embeds_target = environment.get_extra_embeds(
            agent.action_tokenizer,
            agent.critic.action_embed,
            agent.action_embed_target,
        )
        selected_indices: list[int] = []
        q_values: list[float] = []
        done = False
        while not done:
            embeds = environment.update_embeds(
                embeds, agent.critic.action_embed
            )
            embeds_target = environment.update_embeds(
                embeds_target, agent.action_embed_target
            )
            action, q_vector, _ = agent.select_action(
                state,
                embeds["rope"],
                embeds_target["rope"],
                random=False,
                evaluate=True,
            )
            state, _, _, done = environment.step(action)
            selected_indices.append(int(action))
            q_values.append(float(q_vector.max().item()))

        keep = qvalue_keep_count(
            q_values, float(retrieval["stopping_threshold"])
        )
        retained_indices = sorted(selected_indices[:keep])
        outputs.append(
            {
                "row_id": row["row_id"],
                "task": row["task"],
                "question": row["question"],
                "target": row["target"],
                "chunk_count": len(chunks),
                "selected_indices_in_retrieval_order": selected_indices,
                "q_values": q_values,
                "retained_indices_in_document_order": retained_indices,
                "retained_chunks": [chunks[index] for index in retained_indices],
            }
        )
    retrieval_seconds = time.perf_counter() - retrieval_start

    document = {
        "schema_version": 1,
        "experiment": manifest["experiment"],
        "status": manifest["status"],
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - wall_start,
        "task": task,
        "panel": data_path.stem,
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "data": {
            "path": str(data_path),
            "sha256": sha256_file(data_path),
            "rows": len(rows),
        },
        "qrag": {
            "code_commit": manifest["qrag_code"]["commit"],
            "checkpoint": checkpoint_spec,
            "checkpoint_config_sha256": sha256_file(
                checkpoint_config_path
            ),
            "checkpoint_model_sha256": sha256_file(
                checkpoint_dir / checkpoint_spec["model_path"]
            ),
            "base_embedder": base_embedder,
            "max_steps": retrieval["max_steps"],
            "stopping_threshold": retrieval["stopping_threshold"],
            "selection": "greedy",
            "retrieval_only_vllm_import_shim": vllm_import_shim,
            "feedback_adapter": "local_concrete_noop_matching_dummy_intent",
        },
        "retrieval_seconds": retrieval_seconds,
        "retrieval_seconds_per_row": retrieval_seconds / len(rows),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
            "ascent_git_commit": git_value(["rev-parse", "HEAD"]),
            "ascent_git_dirty": bool(git_value(["status", "--porcelain"])),
            "torchdynamo_disabled": os.environ.get("TORCHDYNAMO_DISABLE") == "1",
        },
        "rows": outputs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n")
    print(
        json.dumps(
            {
                "task": task,
                "panel": data_path.stem,
                "rows": len(outputs),
                "mean_retained_chunks": float(
                    np.mean([len(row["retained_chunks"]) for row in outputs])
                ),
                "output": str(output),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--task", choices=("qa2", "qa3"), required=True)
    parser.add_argument("--qrag-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.manifest,
        args.data,
        args.task,
        args.qrag_root,
        args.checkpoint_dir,
        args.output,
    )


if __name__ == "__main__":
    main()
    # Importing Q-RAG creates a non-daemon runtime thread that can keep a
    # completed one-panel process alive indefinitely. All files are already
    # closed here; flush user-visible output, then terminate with success so
    # the GPU allocation is released before the next frozen panel.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
