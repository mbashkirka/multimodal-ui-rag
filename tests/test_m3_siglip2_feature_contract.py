from types import SimpleNamespace

import pytest
import torch

from ui_rag.embeddings.siglip2 import SigLIP2Encoder


def test_pooled_tensor_accepts_transformers_model_output_shape():
    expected = torch.randn(3, 768)
    output = SimpleNamespace(pooler_output=expected)
    actual = SigLIP2Encoder._pooled_tensor(output, modality="image")
    assert actual is expected


def test_pooled_tensor_accepts_direct_tensor_for_forward_compatibility():
    expected = torch.randn(2, 768)
    actual = SigLIP2Encoder._pooled_tensor(expected, modality="text")
    assert actual is expected


def test_pooled_tensor_accepts_return_dict_false_tuple():
    hidden = torch.randn(2, 196, 768)
    pooled = torch.randn(2, 768)
    actual = SigLIP2Encoder._pooled_tensor((hidden, pooled), modality="image")
    assert actual is pooled


def test_pooled_tensor_rejects_missing_pooler_output():
    with pytest.raises(TypeError, match="pooler_output"):
        SigLIP2Encoder._pooled_tensor(SimpleNamespace(), modality="text")


def test_pooled_tensor_rejects_non_matrix_pooler_output():
    with pytest.raises(RuntimeError, match="rank-2"):
        SigLIP2Encoder._pooled_tensor(
            SimpleNamespace(pooler_output=torch.randn(2, 5, 768)),
            modality="image",
        )


def test_siglip2_text_contract_is_explicit():
    assert SigLIP2Encoder.explicit_lowercase is True
    assert SigLIP2Encoder.text_max_length == 64


class _FakeProcessor:
    def __init__(self):
        self.last_kwargs = None

    def __call__(self, **kwargs):
        self.last_kwargs = kwargs
        return {"input_ids": torch.ones((len(kwargs["text"]), 64), dtype=torch.long)}


class _FakeModel:
    def get_text_features(self, **_kwargs):
        return SimpleNamespace(pooler_output=torch.tensor([[3.0, 4.0]]))


def test_encode_text_lowercases_and_unwraps_pooler_output():
    encoder = SigLIP2Encoder.__new__(SigLIP2Encoder)
    encoder.device = "cpu"
    encoder.processor = _FakeProcessor()
    encoder.model = _FakeModel()

    vectors = encoder.encode_text(["DaRk Banking LOGIN Screen"])

    assert encoder.processor.last_kwargs["text"] == ["dark banking login screen"]
    assert encoder.processor.last_kwargs["padding"] == "max_length"
    assert encoder.processor.last_kwargs["truncation"] is True
    assert encoder.processor.last_kwargs["max_length"] == 64
    assert vectors.shape == (1, 2)
    assert vectors.dtype.name == "float32"
    assert vectors[0, 0] == pytest.approx(0.6)
    assert vectors[0, 1] == pytest.approx(0.8)
