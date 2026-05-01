# COMP579 Project MARL

This repository is based on [EIR-MAPPO](https://github.com/DIG-Beihang/EIR-MAPPO) and [HARL](https://github.com/PKU-MARL/HARL).
Use this README as the single run guide.

## 1) Setup

1. Follow HARL/EIR-MAPPO environment setup first (PyTorch + environment-specific dependencies like SMAC/LBF).
2. Install extra Python packages used by this project:

```bash
pip install tensorboardX tensorboard matplotlib pytest pyyaml
```

## 2) Quick Test

```bash
pytest tests/test_attack_prob_logic.py -v
```

## 3) Train (single run)

Use `train.py` and `--algo`:

```bash
python -u train.py --algo mappo_advt_belief --env lbforaging --exp_name baseline --seed 1
```

SMAC example:

```bash
python -u train.py --algo mappo_advt_belief --env smac --exp_name baseline --map_name 4m_vs_3m --seed 1
```

## 4) Batch Experiment (attack_prob)

```bash
bash scripts/run_attack_prob_training.sh lbforaging '' 5000000
```

SMAC example:

```bash
bash scripts/run_attack_prob_training.sh smac 4m_vs_3m 5000000
```

## 5) Aggregate Results

```bash
python scripts/aggregate_attack_prob_results.py --env lbforaging --out-dir analysis/attack_prob_lbf
```

SMAC example:

```bash
python scripts/aggregate_attack_prob_results.py --env smac --map 4m_vs_3m --out-dir analysis/attack_prob_smac_4m_vs_3m
```

## 6) Outputs

Training outputs are under:

- `eir_mappo/results/{env}/{map_name}/mappo_advt_belief/{exp_name}/{seed}/run{n}/`
- For environments without `map_name`, path is `eir_mappo/results/{env}/...`

Main files:

- `models/` model checkpoints
- `logs/` tensorboard/progress logs
- `config.json` run configuration
