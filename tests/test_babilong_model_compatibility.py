from types import SimpleNamespace

import pytest

from experiments.run_babilong_prompt import (
    render_chat_prompt,
    validate_loaded_model,
    validate_runtime_dependencies,
)


class FakeParameter:
    def __init__(self, count: int) -> None:
        self.count = count

    def numel(self) -> int:
        return self.count


class FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            model_type="qwen2",
            architectures=["Qwen2ForCausalLM"],
            max_position_embeddings=32768,
        )

    def parameters(self):
        return iter((FakeParameter(60), FakeParameter(40)))


class RecordingTokenizer:
    def __init__(self) -> None:
        self.call = None

    def apply_chat_template(self, messages, **kwargs):
        self.call = (messages, kwargs)
        return "rendered"


def endpoint(**updates):
    value = {
        "model_parameters": 100,
        "model_type": "qwen2",
        "architecture": "Qwen2ForCausalLM",
    }
    value.update(updates)
    return value


def test_loaded_model_must_match_frozen_identity_and_native_context() -> None:
    identity = validate_loaded_model(FakeModel(), endpoint(), context_tokens=16384)
    assert identity == {
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
        "native_context_tokens": 32768,
        "loaded_parameters": 100,
    }


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"model_type": "llama"}, "model_type mismatch"),
        ({"architecture": "LlamaForCausalLM"}, "architecture mismatch"),
        ({"model_parameters": 99}, "parameter-count mismatch"),
    ],
)
def test_loaded_model_identity_mismatch_fails_closed(updates, message) -> None:
    with pytest.raises(RuntimeError, match=message):
        validate_loaded_model(FakeModel(), endpoint(**updates), context_tokens=16384)


def test_loaded_model_rejects_too_short_native_context() -> None:
    with pytest.raises(RuntimeError, match="native context"):
        validate_loaded_model(FakeModel(), endpoint(), context_tokens=32769)


def test_chat_template_switches_are_endpoint_frozen() -> None:
    tokenizer = RecordingTokenizer()
    rendered = render_chat_prompt(
        tokenizer,
        "question",
        endpoint(chat_template_kwargs={"enable_thinking": False}),
    )
    assert rendered == "rendered"
    messages, kwargs = tokenizer.call
    assert messages == [{"role": "user", "content": "question"}]
    assert kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }


def test_chat_template_kwargs_must_be_mapping() -> None:
    with pytest.raises(RuntimeError, match="string-keyed mapping"):
        render_chat_prompt(
            RecordingTokenizer(),
            "question",
            endpoint(chat_template_kwargs=["enable_thinking", False]),
        )


def test_frozen_system_prompt_is_explicitly_rendered() -> None:
    tokenizer = RecordingTokenizer()
    render_chat_prompt(
        tokenizer,
        "question",
        endpoint(system_prompt="Frozen system text."),
    )
    messages, _ = tokenizer.call
    assert messages == [
        {"role": "system", "content": "Frozen system text."},
        {"role": "user", "content": "question"},
    ]


def test_runtime_dependencies_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        "experiments.run_babilong_prompt.importlib.metadata.version",
        lambda name: {"transformers": "4.57.6"}[name],
    )
    assert validate_runtime_dependencies(
        {"runtime_dependencies": {"transformers": "4.57.6"}}
    ) == {"transformers": "4.57.6"}
    with pytest.raises(RuntimeError, match="runtime dependency mismatch"):
        validate_runtime_dependencies(
            {"runtime_dependencies": {"transformers": "4.56.0"}}
        )
