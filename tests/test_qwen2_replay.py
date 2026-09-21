import pytest

pytest.importorskip("torch")  # optional GPU-stack / audit dependency; skipped in the CPU-only public tree

import torch
import transformers

from ascent.qwen2_replay import replay_from_hidden


def test_qwen2_replay_without_prefix_matches_frozen_suffix() -> None:
    config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = transformers.Qwen2ForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 5, 9, 3]])
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        replayed = replay_from_hidden(
            model,
            output.hidden_states[1],
            attention_mask,
            output.hidden_states[1][:, :0],
            1,
        )
    torch.testing.assert_close(replayed, output.hidden_states[-1])


def test_qwen2_replay_rejects_width_mismatch() -> None:
    config = transformers.Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model = transformers.Qwen2ForCausalLM(config)
    with pytest.raises(ValueError, match="widths"):
        replay_from_hidden(
            model,
            torch.zeros(1, 2, 16),
            torch.ones(1, 2, dtype=torch.long),
            torch.zeros(1, 1, 8),
            0,
        )
