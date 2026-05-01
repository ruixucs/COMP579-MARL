"""Unit tests for the step-level attack mask logic implemented in
``OnPolicyMARunnerAdvtBelief.run`` and ``_eval_adv``.

These tests reproduce the mask math without booting an RL env. They cover:
  1) attack_prob = 1.0 reduces to legacy (per-episode) behavior.
  2) attack_prob = 0.0 disables all attacks.
  3) clean threads (episode_adversary=False) are never attacked.
  4) For all-adversary threads, observed attack rate matches attack_prob.
  5) Joint probability when both layers (episode_adv + step_attack) are stochastic.
  6) The action-override numpy slicing matches the in-codebase pattern.
"""

import numpy as np
import pytest


def step_attack_mask(episode_adversary: np.ndarray, attack_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Mirror of the runner-side logic:
        step_attack = (np.random.rand(n_threads) < self.attack_prob)
        attack_mask = self.episode_adversary & step_attack
    """
    n_threads = episode_adversary.shape[0]
    step_attack = rng.random(n_threads) < attack_prob
    return episode_adversary & step_attack


def test_attack_prob_one_is_equivalent_to_episode_adversary():
    rng = np.random.default_rng(0)
    episode_adv = np.array([True, True, False, False, True])
    mask = step_attack_mask(episode_adv, attack_prob=1.0, rng=rng)
    np.testing.assert_array_equal(mask, episode_adv)


def test_attack_prob_zero_disables_all_attacks():
    rng = np.random.default_rng(0)
    episode_adv = np.array([True, True, False, False, True])
    mask = step_attack_mask(episode_adv, attack_prob=0.0, rng=rng)
    assert not mask.any()


def test_clean_threads_never_attacked_regardless_of_attack_prob():
    rng = np.random.default_rng(123)
    n = 100_000
    episode_adv = np.zeros(n, dtype=bool)
    mask = step_attack_mask(episode_adv, attack_prob=0.8, rng=rng)
    assert not mask.any(), "clean threads must never be attacked"


@pytest.mark.parametrize("attack_prob", [0.2, 0.5, 0.8])
def test_attack_rate_matches_attack_prob_when_episode_adv_all_true(attack_prob):
    rng = np.random.default_rng(seed=42)
    n = 200_000
    episode_adv = np.ones(n, dtype=bool)
    mask = step_attack_mask(episode_adv, attack_prob=attack_prob, rng=rng)
    actual = mask.mean()
    assert abs(actual - attack_prob) < 0.005, f"expected ~{attack_prob}, got {actual}"


def test_attack_rate_is_joint_probability_when_some_clean():
    """If 50% threads are clean and attack_prob=0.5, overall mask rate ≈ 0.25."""
    rng = np.random.default_rng(7)
    n = 200_000
    episode_adv = rng.random(n) < 0.5
    mask = step_attack_mask(episode_adv, attack_prob=0.5, rng=rng)
    assert abs(mask.mean() - 0.25) < 0.01


def test_action_override_only_for_masked_threads():
    """Reproduces input_actions[mask, adv_id] = adv_actions[mask, adv_id]."""
    n_threads, n_agents = 4, 3
    adv_id = 1
    actor_actions = np.zeros((n_threads, n_agents, 1), dtype=np.int64)
    adv_actions = np.full_like(actor_actions, fill_value=9)
    mask = np.array([True, False, True, False])

    out = actor_actions.copy()
    out[mask, adv_id] = adv_actions[mask, adv_id]

    expected = np.array(
        [[[0], [9], [0]],
         [[0], [0], [0]],
         [[0], [9], [0]],
         [[0], [0], [0]]],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(out, expected)
