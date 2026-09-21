#!/usr/bin/env python3
"""Posthoc, label-free audit of BABILong decode context sensitivity."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ascent.babilong_memory import read_babilong
from experiments.run_babilong_prompt import answer_instruction


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def build_ascent_prompt(
    row: dict[str, Any], fact_slots: int, *, strip_question: bool = False
) -> str:
    read = read_babilong(row["input"], row["question"])
    memory_text = "\n".join(f"- {fact.source}" for fact in read.facts[-fact_slots:])
    question = row["question"].strip() if strip_question else row["question"]
    return (
        "A bounded causal memory retained the following relevant facts in "
        "chronological order. Answer using only these facts. "
        f"{answer_instruction(row['task'])}\n\n"
        f"Memory:\n{memory_text}\n\nQuestion: {question}\nAnswer:"
    )


def build_operator_prompt(row: dict[str, Any], fact_slots: int) -> str:
    """Build a target-blind operator prompt without leading null-label priming."""
    read = read_babilong(row["input"], row["question"])
    memory_text = "\n".join(f"- {fact.source}" for fact in read.facts[-fact_slots:])
    update_rules = (
        "Start from an empty inventory and apply every retained event in order. "
        "An object enters the inventory when the queried person gets, grabs, "
        "takes, picks up, or receives it after 'to'. It leaves when that person "
        "leaves, discards, drops, puts down, gives, hands, or passes it to "
        "someone else."
    )
    if row["task"] == "qa7":
        output_rule = (
            "Count the objects after the final event. Reply with only the count "
            "as one lowercase word."
        )
    elif row["task"] == "qa8":
        output_rule = (
            "Reply with only the lowercase object names in acquisition order, "
            "separated by commas. If the final inventory is empty, reply nothing."
        )
    else:
        raise ValueError("operator prompt is restricted to QA7/QA8")
    return (
        f"{update_rules} {output_rule}\n\n"
        f"Events:\n{memory_text}\n\nQuestion: {row['question'].strip()}\nAnswer:"
    )


def run(
    config_path: Path,
    data_path: Path,
    endpoint_name: str,
    model_dir: Path,
    output: Path,
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = json.loads(config_path.read_text())
    endpoint = {row["name"]: row for row in config["endpoints"]}[endpoint_name]
    rows = [json.loads(line) for line in data_path.read_text().splitlines()]
    prompts = [
        build_ascent_prompt(row, int(endpoint["state_fact_slots"])) for row in rows
    ]
    stripped_prompts = [
        build_ascent_prompt(row, int(endpoint["state_fact_slots"]), strip_question=True)
        for row in rows
    ]
    operator_prompts = [
        build_operator_prompt(row, int(endpoint["state_fact_slots"])) for row in rows
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, dtype=torch.bfloat16
        )
        .to("cuda")
        .eval()
    )
    torch.manual_seed(int(config["seed"]))

    def decode(
        indices: list[int], *, prompt_pool: list[str] = prompts
    ) -> dict[str, Any]:
        selected = [prompt_pool[index] for index in indices]
        chat_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in selected
        ]
        encoded = tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(config["context_tokens"]),
            add_special_tokens=False,
        ).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=int(config["max_new_tokens"]),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        suffix = generated[:, encoded.input_ids.shape[1] :]
        return {
            "indices": indices,
            "row_ids": [rows[index]["row_id"] for index in indices],
            "prompt_sha256": [prompt_sha256(prompt) for prompt in selected],
            "input_ids_sha256": [
                hashlib.sha256(
                    encoded.input_ids[row_index].detach().cpu().numpy().tobytes()
                ).hexdigest()
                for row_index in range(len(indices))
            ],
            "outputs": tokenizer.batch_decode(suffix, skip_special_tokens=True),
        }

    batch_size = int(endpoint["batch_size"])
    target_indices = [22, 23]
    operator_outputs: list[str] = []
    for start in range(0, len(operator_prompts), batch_size):
        record = decode(
            list(range(start, min(start + batch_size, len(operator_prompts)))),
            prompt_pool=operator_prompts,
        )
        operator_outputs.extend(record["outputs"])
    stripped_target_pair_process_cold = decode(
        target_indices, prompt_pool=stripped_prompts
    )
    stripped_outputs: list[str] = []
    for start in range(0, len(stripped_prompts), batch_size):
        record = decode(
            list(range(start, min(start + batch_size, len(stripped_prompts)))),
            prompt_pool=stripped_prompts,
        )
        stripped_outputs.extend(record["outputs"])
    records = {
        "operator_prompt_first_sequence": {
            "status": "posthoc_target_blind_operator_prompt_no_promotion",
            "indices": list(range(len(operator_prompts))),
            "prompt_sha256": [prompt_sha256(prompt) for prompt in operator_prompts],
            "outputs": operator_outputs,
            "target_pair_outputs": [
                operator_outputs[index] for index in target_indices
            ],
        },
        "stripped_target_pair_process_cold": stripped_target_pair_process_cold,
        "stripped_question_first_sequence": {
            "normalization": "Python str.strip on the question field only",
            "indices": list(range(len(stripped_prompts))),
            "prompt_sha256": [prompt_sha256(prompt) for prompt in stripped_prompts],
            "outputs": stripped_outputs,
            "target_pair_outputs": [
                stripped_outputs[index] for index in target_indices
            ],
        },
        "target_pair_cold": decode(target_indices),
    }
    full_outputs: list[str] = []
    for start in range(0, len(prompts), batch_size):
        record = decode(list(range(start, min(start + batch_size, len(prompts)))))
        full_outputs.extend(record["outputs"])
    records["full_sequence"] = {
        "indices": list(range(len(prompts))),
        "outputs": full_outputs,
        "target_pair_outputs": [full_outputs[index] for index in target_indices],
    }
    records["target_pair_after_full"] = decode(target_indices)
    records["qa8_only_then_target"] = decode(list(range(20, len(prompts))))
    records["target_pair_after_qa8"] = decode(target_indices)
    stripped_outputs = []
    for start in range(0, len(stripped_prompts), batch_size):
        record = decode(
            list(range(start, min(start + batch_size, len(stripped_prompts)))),
            prompt_pool=stripped_prompts,
        )
        stripped_outputs.extend(record["outputs"])
    records["stripped_question_full_sequence"] = {
        "normalization": "Python str.strip on the question field only",
        "indices": list(range(len(stripped_prompts))),
        "prompt_sha256": [prompt_sha256(prompt) for prompt in stripped_prompts],
        "outputs": stripped_outputs,
        "target_pair_outputs": [stripped_outputs[index] for index in target_indices],
    }
    result = {
        "status": "posthoc_label_free_decode_context_diagnostic_no_promotion",
        "endpoint": endpoint_name,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.data, args.endpoint, args.model_dir, args.output)


if __name__ == "__main__":
    main()
