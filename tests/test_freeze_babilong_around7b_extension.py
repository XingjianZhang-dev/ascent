import copy
import json
from pathlib import Path

import pytest

from experiments.freeze_babilong_around7b_extension import build_config
from experiments.run_natural_repeat import sha256_file


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "configs/babilong_around7b_16k_extension_design.json"
SOURCE = ROOT / "configs/babilong_qwen2p5_canonical_coscale_16k_confirmatory.json"


def _audits(tmp_path: Path) -> list[Path]:
    design = json.loads(DESIGN.read_text())
    paths = []
    for index, model in enumerate(design["frozen_model_roster"]):
        document = {
            "status": "score_free_checkpoint_and_template_audit",
            "model": model,
            "design": {"sha256": sha256_file(DESIGN)},
            "model_config": {
                "model_type": f"type-{index}",
                "architectures": [f"Architecture{index}"],
                "native_context_tokens": 32768,
            },
            "weights": {
                "parameters": 7_000_000_000 + index,
                "files": [{
                    "path": f"model-{index}.safetensors",
                    "bytes": 100 + index,
                    "sha256": f"{index:064x}",
                }],
            },
            "chat_template": {"rendered_probe_sha256": f"{index + 10:064x}"},
            "environment": {
                "git_dirty": False,
                "libraries": {
                    "transformers": "4.57.6",
                    "tokenizers": "0.22.2",
                    "sentencepiece": "0.2.2",
                    "safetensors": "0.6.2",
                },
            },
            "decoder_scores_observed": False,
            "audit_pass": True,
            "official_lfs_manifest_exact_match": True,
            "official_critical_manifest_exact_match": True,
            "safetensors_index_validation": {
                "tensor_count_match": True,
                "resolved_by_official_index_and_exact_model_load": True,
            },
            "model_load_validation": {
                "load_pass": True,
                "generation_invoked": False,
            },
        }
        path = tmp_path / f"audit-{index}.json"
        path.write_text(json.dumps(document))
        paths.append(path)
    return paths


def test_build_config_is_authorized_only_from_all_score_free_audits(tmp_path):
    audits = _audits(tmp_path)
    config = build_config(
        DESIGN, SOURCE, audits, frozen_utc="2026-08-15T04:00:00+08:00"
    )
    assert config["execution_authorization"]["authorized"] is True
    assert len(config["endpoints"]) == 5
    assert set(config["conditions"]["canonical_slots_4"]["state_fact_slots_by_endpoint"].values()) == {4}
    assert config["conditions"]["canonical_fixed_slots_3"]["state_fact_slots_by_endpoint"] == {"qwen2p5-7b-instruct": 3}
    assert config["endpoints"][-1]["chat_template_kwargs"] == {"enable_thinking": False}
    granite = next(row for row in config["endpoints"] if row["name"] == "granite-3p3-8b-instruct")
    assert "August 15, 2026" in granite["system_prompt"]
    assert config["runtime_dependencies"]["sentencepiece"] == "0.2.2"


@pytest.mark.parametrize("mutation", ["score_seen", "dirty", "failed"])
def test_build_config_fails_closed(tmp_path, mutation):
    audits = _audits(tmp_path)
    document = json.loads(audits[0].read_text())
    if mutation == "score_seen":
        document["decoder_scores_observed"] = True
    elif mutation == "dirty":
        document["environment"]["git_dirty"] = True
    else:
        document["audit_pass"] = False
    audits[0].write_text(json.dumps(document))
    with pytest.raises(RuntimeError):
        build_config(DESIGN, SOURCE, audits, frozen_utc="x")
