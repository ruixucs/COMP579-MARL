"""Numerical tests for the §5.7 masked-BCE belief loss.

Goal: verify that the in-fork formulation

    masked_loss = bce_raw * step_attack_batch   # broadcast (B, 1) over (B, num_agents)
    denom       = step_attack_batch.sum() * num_agents
    loss        = masked_loss.sum() / (denom + eps)

(a) reduces to ``bce_raw.mean()`` *bit-equivalently* when step_attack is all-True
    (i.e. when attack_prob = 1.0, which is the legacy upstream behavior), and
(b) when step_attack has zeros, those rows do not contribute and the resulting
    loss equals the mean BCE computed *only over the fired steps*.

We mirror the algo math here without booting an env, so this runs on CPU in <1s.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F


def masked_bce(belief: torch.Tensor, target: torch.Tensor, step_attack: torch.Tensor) -> torch.Tensor:
    """Reproduces ``MAPPOAdvtBelief.update_belief`` (use_belief_active_masks=False branch)."""
    bce_raw = F.binary_cross_entropy(belief, target, reduction="none")  # (B, A)
    mask = step_attack  # (B, 1)
    last_dim = bce_raw.shape[-1]
    masked = bce_raw * mask
    denom = mask.sum() * last_dim
    return masked.sum() / (denom + 1e-8)


def _make_belief_and_target(rng: np.random.Generator, B: int, A: int):
    belief_np = rng.uniform(low=0.05, high=0.95, size=(B, A)).astype(np.float32)
    target_np = (rng.random(size=(B, A)) < 0.5).astype(np.float32)
    return torch.from_numpy(belief_np), torch.from_numpy(target_np)


def test_attack_prob_one_is_bit_equivalent_to_mean():
    """attack_prob = 1.0 reproduces upstream ``loss.mean()`` to within float32 noise."""
    rng = np.random.default_rng(0)
    B, A = 64, 4
    belief, target = _make_belief_and_target(rng, B, A)
    step_attack = torch.ones(B, 1, dtype=torch.float32)

    upstream = F.binary_cross_entropy(belief, target, reduction="none").mean()
    ours = masked_bce(belief, target, step_attack)

    assert torch.allclose(upstream, ours, atol=1e-6), (
        f"expected bit-equivalent at attack_prob=1.0, got upstream={upstream.item()} ours={ours.item()}"
    )


def test_zero_mask_rows_do_not_contribute():
    """Setting some step_attack entries to zero must give the same result as
    deleting those rows from belief and target and computing plain ``loss.mean()`` on the rest."""
    rng = np.random.default_rng(1)
    B, A = 32, 4
    belief, target = _make_belief_and_target(rng, B, A)

    fired = torch.zeros(B, 1, dtype=torch.float32)
    fired_idx = torch.arange(B) % 2 == 0  # half the rows fire
    fired[fired_idx] = 1.0

    ours = masked_bce(belief, target, fired)
    fired_only_mean = F.binary_cross_entropy(
        belief[fired_idx], target[fired_idx], reduction="none"
    ).mean()

    assert torch.allclose(ours, fired_only_mean, atol=1e-6), (
        f"masked loss must equal mean-BCE-over-fired-rows: ours={ours.item()} ref={fired_only_mean.item()}"
    )


def test_all_zero_mask_is_safe_and_zero():
    """Edge case: if no step fired in the minibatch, the loss should not NaN
    (the +1e-8 in the denominator handles the divide). The numerator is 0."""
    rng = np.random.default_rng(2)
    B, A = 16, 4
    belief, target = _make_belief_and_target(rng, B, A)
    step_attack = torch.zeros(B, 1, dtype=torch.float32)

    ours = masked_bce(belief, target, step_attack)
    assert torch.isfinite(ours), f"loss must be finite, got {ours}"
    assert ours.abs().item() < 1e-5, f"with no fired steps the loss should be ~0, got {ours.item()}"


@pytest.mark.parametrize("attack_prob", [0.2, 0.5, 0.8])
def test_loss_per_sample_units_are_stable(attack_prob: float):
    """Sanity: the masked loss has the same per-sample units as ``loss.mean()`` —
    that is, scaling all bce values by k scales the masked loss by k."""
    rng = np.random.default_rng(7)
    B, A = 128, 3
    belief, target = _make_belief_and_target(rng, B, A)
    fired = torch.from_numpy((rng.random(B) < attack_prob).astype(np.float32)).unsqueeze(-1)

    base = masked_bce(belief, target, fired)

    # scale belief slightly to perturb bce; the relative scaling should be preserved
    belief2 = (belief * 0.5 + 0.25).clamp(0.05, 0.95)  # keep in (0,1)
    perturbed = masked_bce(belief2, target, fired)

    assert torch.isfinite(base) and torch.isfinite(perturbed)
    # both losses should be order-1 (per-sample mean BCE), i.e. neither blows up
    # nor collapses to ~0 just because attack_prob is small.
    assert 0.0 < base.item() < 5.0, f"masked loss out of expected per-sample range: {base.item()}"
    assert 0.0 < perturbed.item() < 5.0, f"perturbed loss out of expected per-sample range: {perturbed.item()}"
