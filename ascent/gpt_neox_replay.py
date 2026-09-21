"""Frozen GPT-NeoX latent replay at a registered relative depth."""

from __future__ import annotations

import math
from typing import Any


def relative_layer_index(num_layers: int, rho: float) -> int:
    """Return a valid suffix-start layer using the preregistered floor rule."""
    if num_layers < 2:
        raise ValueError("latent replay requires at least two layers")
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must lie strictly between zero and one")
    return min(num_layers - 1, max(1, math.floor(rho * num_layers)))


def replay_from_hidden(
    model: Any,
    query_hidden: Any,
    query_attention_mask: Any,
    replay_latent: Any,
    layer_index: int,
) -> Any:
    """Prepend replay latents at ``layer_index`` and execute the frozen suffix.

    ``query_hidden`` is the input to the selected GPT-NeoX layer, obtained from
    ``output_hidden_states``. ``replay_latent`` has shape ``[batch, replay, d]``.
    The persistent state is whatever generated the replay latent; this function
    only materializes the transient model-width activation used by the suffix.
    """
    import torch
    from transformers.models.gpt_neox.modeling_gpt_neox import create_causal_mask

    backbone = model.gpt_neox
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
    sequence_length = hidden_states.shape[1]
    cache_position = torch.arange(sequence_length, device=hidden_states.device)
    position_ids = cache_position.unsqueeze(0)
    causal_mask = create_causal_mask(
        config=backbone.config,
        input_embeds=hidden_states,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=position_ids,
    )
    position_embeddings = backbone.rotary_emb(hidden_states, position_ids)
    for layer in backbone.layers[layer_index:]:
        hidden_states = layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            head_mask=None,
            layer_past=None,
            use_cache=False,
            output_attentions=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )[0]
    return backbone.final_layer_norm(hidden_states)
