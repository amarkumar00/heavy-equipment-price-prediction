from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.train_pytorch import PriceMLP


def test_mlp_output_shape_and_round_trip(tmp_path: Path) -> None:
    model = PriceMLP(input_features=7)
    inputs = torch.randn(5, 7)
    outputs = model(inputs)
    assert outputs.shape == (5,)

    artifact = tmp_path / "model.pt"
    torch.save({"state_dict": model.state_dict(), "input_features": 7}, artifact)
    loaded = torch.load(artifact, weights_only=True)
    restored = PriceMLP(input_features=loaded["input_features"])
    restored.load_state_dict(loaded["state_dict"])
    restored.eval()
    with torch.no_grad():
        assert torch.allclose(model.eval()(inputs), restored(inputs))
