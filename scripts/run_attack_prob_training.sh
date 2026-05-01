#!/usr/bin/env bash
# Train EIR-MAPPO defenders under three step-level attack probabilities
# (the adversary's identity is fixed; only its decision to attack at each step
# is probabilistic) across multiple seeds.
#
# COMP579 project: probabilistic adversary experiment.
#
# Usage:
#   bash scripts/run_attack_prob_training.sh <ENV> <MAP> [N_STEPS]
#
# Examples:
#   bash scripts/run_attack_prob_training.sh smac 4m_vs_3m 5000000
#   bash scripts/run_attack_prob_training.sh toy ''       2000000
#   bash scripts/run_attack_prob_training.sh lbforaging '' 5000000
#
# After training, results land in:
#   eir_mappo/results/<env>/[<map>/]mappo_advt_belief/attack_prob_<X>/<seed>/run<n>/

set -euo pipefail

ENV="${1:?usage: <ENV> <MAP> [N_STEPS]}"
MAP="${2-}"
N_STEPS="${3:-5000000}"

PROBS=(0.2 0.5 0.8)
SEEDS=(1 2 3)

cd "$(dirname "$0")/.."

for PROB in "${PROBS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    EXP_NAME="attack_prob_${PROB}"
    echo "================================================================"
    echo "[train] env=${ENV} map=${MAP:-N/A} attack_prob=${PROB} seed=${SEED}"
    echo "================================================================"
    CMD=(python -u train.py
      --algo mappo_advt_belief
      --env "${ENV}"
      --exp_name "${EXP_NAME}"
      --seed "${SEED}"
      --attack_prob "${PROB}"
      --num_env_steps "${N_STEPS}")
    if [ -n "${MAP}" ]; then
      CMD+=(--map_name "${MAP}")
    fi
    "${CMD[@]}"
  done
done

echo "All step-level attack-probability training runs finished."
echo "Aggregate with: python scripts/aggregate_attack_prob_results.py --env ${ENV} --map '${MAP}'"
