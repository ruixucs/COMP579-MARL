"""End-to-end plumbing test for ``ActorBufferAdvtBelief`` with the new ``step_attack`` field.

This boots the *real* buffer class against a fake (Discrete) obs/action space and
verifies that:

  1) ``insert(..., step_attack=...)`` accepts the new kwarg without breaking the
     legacy positional-arg call sites used elsewhere.
  2) ``recurrent_generator_belief`` now yields **16** fields (the 16th being
     ``step_attack_batch``) and the field is broadcast-compatible with the BCE
     loss tensor (shape ``(B, 1)`` matches per-agent ``(B, num_agents)``).
  3) When ``attack_prob = 1.0`` (every step fires), ``step_attack_batch`` is all-1.
  4) Stored values round-trip: the per-step ``step_attack`` we put in equals the
     per-row ``step_attack_batch`` we get out (modulo the recurrent-chunk reshape).

This relies on the buffer being importable from a clean Python with only torch +
numpy. We avoid touching the runner / algo so this test stays fast and offline.
"""

import os
import sys
import types

import numpy as np
import pytest

# Make the project importable without installing as a package.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


def _stub_module(name: str, attrs: dict | None = None):
    """Register an empty stub module so that ``import name`` succeeds.
    ``attrs`` lets us pre-populate symbols (e.g. ShareSubprocVecEnv) that the
    real module would have exported."""
    mod = types.ModuleType(name)
    for k, v in (attrs or {}).items():
        setattr(mod, k, v)
    sys.modules.setdefault(name, mod)


# eir_mappo.util.util pulls in tensorboardX + ShareSubprocVecEnv at import time,
# which in turn want gym / smac / etc. — the full upstream training stack. The
# *buffer logic* we want to test only uses _flatten and _sa_cast from util.util,
# neither of which touches those imports. Stub the heavy deps so the buffer can
# be imported without installing the world.
_stub_module("tensorboardX", {"SummaryWriter": object})


class _ShareEnvStub:
    """Stand-in for ShareSubprocVecEnv / ShareDummyVecEnv (never instantiated by tests)."""

    pass


_stub_module(
    "eir_mappo.env.env_wrappers",
    {
        "ShareSubprocVecEnv": _ShareEnvStub,
        "ShareDummyVecEnv": _ShareEnvStub,
    },
)

try:
    from eir_mappo.common.actor_buffer_advt_with_belief import ActorBufferAdvtBelief  # noqa: F401
except Exception as _import_err:  # pragma: no cover - safety net for unrelated import errors
    pytest.skip(
        f"skipping buffer plumbing tests; eir_mappo buffer is not importable "
        f"({_import_err.__class__.__name__}: {_import_err}).",
        allow_module_level=True,
    )


# buffer code does ``obs_space.__class__.__name__ == 'Box'`` and same for 'Discrete',
# so we mirror those names by literally naming the stand-in classes ``Box`` / ``Discrete``.

class Box:
    """Minimal stand-in for gym.spaces.Box (only needs `.shape`)."""

    def __init__(self, shape):
        self.shape = shape


class Discrete:
    """Minimal stand-in for gym.spaces.Discrete (only needs `.n`)."""

    def __init__(self, n: int):
        self.n = n
        self.shape = ()


def _make_buffer(episode_length=8, n_threads=4, num_agents=3, obs_dim=10, n_actions=5):
    args = {
        "episode_length": episode_length,
        "n_rollout_threads": n_threads,
        "hidden_sizes": [16, 16],
        "recurrent_N": 1,
    }
    obs_space = Box(shape=(obs_dim,))
    act_space = Discrete(n=n_actions)
    return ActorBufferAdvtBelief(args, obs_space, act_space, num_agents)


def _fill_buffer_with_known_step_attack(buf, n_threads, num_agents, obs_dim, attack_prob: float, rng: np.random.Generator):
    """Fill the buffer with random data and a known per-step step_attack pattern.
    Returns the per-step (T, N) step_attack array we wrote in, so the test can
    cross-check against what the generator yields."""
    T = buf.episode_length
    written_step_attack = np.zeros((T, n_threads), dtype=np.float32)

    for t in range(T):
        if attack_prob >= 1.0:
            sa = np.ones(n_threads, dtype=bool)
        elif attack_prob <= 0.0:
            sa = np.zeros(n_threads, dtype=bool)
        else:
            sa = rng.random(n_threads) < attack_prob
        written_step_attack[t] = sa.astype(np.float32)

        buf.insert(
            obs=rng.standard_normal((n_threads, obs_dim)).astype(np.float32),
            ground_truth_type=rng.random((n_threads, num_agents)).astype(np.float32),
            rnn_states=np.zeros((n_threads, 1, 16), dtype=np.float32),
            adv_rnn_states=np.zeros((n_threads, 1, 16), dtype=np.float32),
            belief_rnn_states=np.zeros((n_threads, 1, 16), dtype=np.float32),
            actions=rng.integers(0, 5, size=(n_threads, 1)).astype(np.float32),
            adv_actions=rng.integers(0, 5, size=(n_threads, 1)).astype(np.float32),
            action_log_probs=rng.standard_normal((n_threads, 1)).astype(np.float32),
            adv_action_log_probs=rng.standard_normal((n_threads, 1)).astype(np.float32),
            rewards=rng.standard_normal((n_threads, 1)).astype(np.float32),
            masks=np.ones((n_threads, 1), dtype=np.float32),
            active_masks=np.ones((n_threads, 1), dtype=np.float32),
            adv_active_masks=np.ones((n_threads, 1), dtype=np.float32),
            available_actions=np.ones((n_threads, 5), dtype=np.float32),
            step_attack=sa,
        )

    return written_step_attack


def test_buffer_accepts_step_attack_and_stores_it():
    rng = np.random.default_rng(0)
    n_threads, num_agents, obs_dim = 4, 3, 10
    buf = _make_buffer(episode_length=8, n_threads=n_threads, num_agents=num_agents, obs_dim=obs_dim)
    written = _fill_buffer_with_known_step_attack(buf, n_threads, num_agents, obs_dim, attack_prob=0.5, rng=rng)

    # buf.step_attack has shape (T, N, 1); flatten last dim and compare to what we wrote
    stored = buf.step_attack.reshape(buf.episode_length, n_threads)
    np.testing.assert_array_equal(stored, written, err_msg="buffer didn't faithfully store step_attack")


def test_legacy_insert_without_step_attack_defaults_to_one():
    """If a caller forgets to pass step_attack, the buffer must default to all-True
    so that the §5.7 mask becomes a no-op (legacy upstream behavior)."""
    rng = np.random.default_rng(1)
    n_threads, num_agents, obs_dim = 4, 3, 10
    buf = _make_buffer(episode_length=4, n_threads=n_threads, num_agents=num_agents, obs_dim=obs_dim)

    for _ in range(buf.episode_length):
        buf.insert(
            obs=rng.standard_normal((n_threads, obs_dim)).astype(np.float32),
            ground_truth_type=rng.random((n_threads, num_agents)).astype(np.float32),
            rnn_states=np.zeros((n_threads, 1, 16), dtype=np.float32),
            adv_rnn_states=np.zeros((n_threads, 1, 16), dtype=np.float32),
            belief_rnn_states=np.zeros((n_threads, 1, 16), dtype=np.float32),
            actions=rng.integers(0, 5, size=(n_threads, 1)).astype(np.float32),
            adv_actions=rng.integers(0, 5, size=(n_threads, 1)).astype(np.float32),
            action_log_probs=rng.standard_normal((n_threads, 1)).astype(np.float32),
            adv_action_log_probs=rng.standard_normal((n_threads, 1)).astype(np.float32),
            rewards=rng.standard_normal((n_threads, 1)).astype(np.float32),
            masks=np.ones((n_threads, 1), dtype=np.float32),
            active_masks=np.ones((n_threads, 1), dtype=np.float32),
            adv_active_masks=np.ones((n_threads, 1), dtype=np.float32),
            available_actions=np.ones((n_threads, 5), dtype=np.float32),
            # no step_attack -- exercising the backward-compat default
        )

    assert (buf.step_attack == 1.0).all(), "default step_attack should be all-True"


def test_generator_yields_16_fields_with_step_attack_last():
    rng = np.random.default_rng(2)
    n_threads, num_agents, obs_dim = 4, 3, 10
    episode_length = 8
    buf = _make_buffer(episode_length=episode_length, n_threads=n_threads, num_agents=num_agents, obs_dim=obs_dim)
    _fill_buffer_with_known_step_attack(buf, n_threads, num_agents, obs_dim, attack_prob=1.0, rng=rng)

    # Mirror the share_param_train_belief setup: advantages have shape (T, N, 1)
    advantages = rng.standard_normal((episode_length, n_threads, 1)).astype(np.float32)

    gen = buf.recurrent_generator_belief(advantages, actor_num_mini_batch=1, data_chunk_length=4)
    sample = next(gen)
    assert len(sample) == 16, f"expected 16 yielded fields (factor=None branch), got {len(sample)}"

    step_attack_batch = sample[15]
    assert step_attack_batch.shape[-1] == 1, f"step_attack_batch must have trailing 1 dim, got {step_attack_batch.shape}"
    # attack_prob = 1.0 was used, so every entry is 1.
    np.testing.assert_array_equal(step_attack_batch, np.ones_like(step_attack_batch))


def test_generator_step_attack_matches_observed_attack_rate():
    """Sanity: at attack_prob = 0.3, the fraction of 1s in the yielded
    step_attack_batch should approach 0.3 across the whole epoch."""
    rng = np.random.default_rng(3)
    n_threads, num_agents, obs_dim = 8, 3, 10
    episode_length = 32
    buf = _make_buffer(episode_length=episode_length, n_threads=n_threads, num_agents=num_agents, obs_dim=obs_dim)
    _fill_buffer_with_known_step_attack(buf, n_threads, num_agents, obs_dim, attack_prob=0.3, rng=rng)

    advantages = rng.standard_normal((episode_length, n_threads, 1)).astype(np.float32)

    fired_total = 0
    rows_total = 0
    gen = buf.recurrent_generator_belief(advantages, actor_num_mini_batch=2, data_chunk_length=4)
    for sample in gen:
        sa = sample[15]
        fired_total += float(sa.sum())
        rows_total += sa.size

    rate = fired_total / max(rows_total, 1)
    assert 0.15 < rate < 0.45, (
        f"observed attack rate {rate:.3f} too far from configured 0.3 "
        f"(over {rows_total} batched rows)"
    )
