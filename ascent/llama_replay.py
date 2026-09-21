"""Frozen Llama latent replay at a registered decoder depth."""

from __future__ import annotations

from typing import Any


def replay_from_hidden(
    model: Any,
    query_hidden: Any,
    query_attention_mask: Any,
    replay_latent: Any,
    layer_index: int,
) -> Any:
    """Prepend replay latents and execute the remaining frozen Llama layers."""
    import inspect

    import torch

    backbone = model.model
    if query_hidden.ndim != 3 or replay_latent.ndim != 3:
        raise ValueError("query and replay tensors must be rank three")
    if query_hidden.shape[0] != replay_latent.shape[0]:
        raise ValueError("query and replay batch sizes must match")
    if query_hidden.shape[2] != replay_latent.shape[2]:
        raise ValueError("query and replay widths must match")
    if not 0 <= layer_index < len(backbone.layers):
        raise ValueError("layer_index is outside the model")

    replay_count = replay_latent.shape[1]
    hidden_states = torch.cat([replay_latent, query_hidden], dim=1)
    replay_mask = torch.ones(
        query_attention_mask.shape[0],
        replay_count,
        dtype=query_attention_mask.dtype,
        device=query_attention_mask.device,
    )
    attention_mask = torch.cat([replay_mask, query_attention_mask], dim=1)
    cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
    position_ids = cache_position.unsqueeze(0)
    try:
        from transformers.masking_utils import create_causal_mask

        causal_mask = create_causal_mask(
            config=backbone.config,
            input_embeds=hidden_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )
    except ImportError:
        causal_mask = backbone._update_causal_mask(
            attention_mask,
            hidden_states,
            cache_position,
            None,
            False,
        )
    position_embeddings = backbone.rotary_emb(hidden_states, position_ids)
    for layer in backbone.layers[layer_index:]:
        layer_parameters = inspect.signature(layer.forward).parameters
        past_key_name = (
            "past_key_values"
            if "past_key_values" in layer_parameters
            else "past_key_value"
        )
        layer_output = layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **{past_key_name: None},
        )
        hidden_states = (
            layer_output[0] if isinstance(layer_output, tuple) else layer_output
        )
    return backbone.norm(hidden_states)
