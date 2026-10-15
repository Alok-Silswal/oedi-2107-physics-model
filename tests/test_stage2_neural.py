import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import train_stage2_neural as neural


def test_uncentered_target_scale_and_signed_power_reconciliation():
    labels = np.array([4., 6., 8.])
    scale = neural.correction_scale(labels)
    assert scale == np.std(labels, ddof=0)
    assert (labels/scale).mean() > 0
    physics = np.array([0., 100., 50.])
    correction = np.array([-2., 5., -70.])
    final = neural.final_power(physics, correction/scale, scale)
    np.testing.assert_allclose(final-physics, correction)
    assert final[0] < 0 and final[-1] < 0
    with pytest.raises(ValueError, match="standard deviation"):
        neural.correction_scale(np.ones(3))


def test_preparation_reports_missing_dependencies_without_training_or_writes(monkeypatch, tmp_path):
    spec = dict(dimension=14, counts={}, target_scale_kw=2.,
                versions={p: None for p in ("torch", "pennylane", "pennylane-lightning")})
    monkeypatch.setattr(neural, "prepare_data", lambda: ({}, {}, np.array([True]), spec, {}))
    def forbidden(*args, **kwargs):
        raise AssertionError("Preparation must not construct/update models or write artifacts")
    for name in ("make_model", "pilot", "train", "freeze", "checkpoint"):
        monkeypatch.setattr(neural, name, forbidden)
    monkeypatch.setattr(neural, "OUT", tmp_path/"experiment")
    monkeypatch.setattr(sys, "argv", ["train_stage2_neural.py", "--prepare"])
    neural.main()
    assert not list(tmp_path.iterdir())


def test_changed_eligibility_is_reported_before_preprocessing(monkeypatch):
    tables = {"validation": pd.DataFrame({"target_timestamp": pd.date_range("2022-02-01", periods=2)})}
    monkeypatch.setattr(neural, "load_examples", lambda: ({"training": [0, 1], "validation": [0, 1]}, tables, {}, []))
    monkeypatch.setattr(neural, "solar_elevation", lambda index: pd.Series(1., index=index))
    with pytest.raises(ValueError, match=r"Eligibility differs: \(2, 2, 2\)"):
        neural.prepare_data()


@pytest.mark.parametrize("kind", ["classical", "quantum"])
def test_shapes_gradients_counts_and_identical_shared_initialization(kind):
    torch = pytest.importorskip("torch")
    if kind == "quantum":
        pytest.importorskip("pennylane")
        pytest.importorskip("pennylane_lightning")
    torch.set_num_threads(4)
    model = neural.make_model(kind, 14, 2107)
    assert all(p.dtype == torch.float64 for p in model.parameters())
    assert model(torch.ones(3, 14, dtype=torch.float64)).dtype == torch.float64
    assert neural.gradient_check(model, 14) == (85 if kind=="classical" else 89)
    shared = neural.make_model("classical", 14, 2107)
    for name in ("projection", "head"):
        for actual, expected in zip(getattr(model, name).parameters(), getattr(shared, name).parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if kind == "quantum":
        assert model.weights.shape == (2, 4, 3)
        assert torch.all(model.weights.abs() <= 0.1)


def test_checkpoint_roundtrip_restores_optimizer_and_random_order_rejects_incompatibility(tmp_path):
    torch = pytest.importorskip("torch")
    model = neural.make_model("classical", 14, 2107)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    x = torch.ones(2, 14, dtype=torch.float64)
    torch.nn.HuberLoss()(model(x), torch.tensor([1., -1.], dtype=torch.float64)).backward()
    optimizer.step()
    generator = torch.Generator().manual_seed(2107)
    torch.randperm(17, generator=generator)
    spec = dict(seed=2107, features=neural.FEATURES, dimension=14)
    path = tmp_path/"last.pt"
    neural.checkpoint(path, dict(specification=spec, model=model.state_dict(), optimizer=optimizer.state_dict(),
                                 epoch=1, shuffle_rng=generator.get_state()))
    expected_order = torch.randperm(17, generator=generator)
    state = neural.read_checkpoint(path, spec)
    restored = neural.make_model("classical", 14, 2107)
    restored.load_state_dict(state["model"])
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.001)
    restored_optimizer.load_state_dict(state["optimizer"])
    torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)
    for current, opt in ((model, optimizer), (restored, restored_optimizer)):
        opt.zero_grad(set_to_none=True)
        torch.nn.HuberLoss()(current(x), torch.tensor([1., -1.], dtype=torch.float64)).backward()
        opt.step()
    torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)
    generator.set_state(state["shuffle_rng"])
    torch.testing.assert_close(expected_order, torch.randperm(17, generator=generator), rtol=0, atol=0)
    with pytest.raises(ValueError, match="Incompatible resume"):
        neural.read_checkpoint(path, dict(spec, dimension=7))
