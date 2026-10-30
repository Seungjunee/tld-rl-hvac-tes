# TLD-RL: Thermal-Load-Decoupled Reinforcement Learning for Cross-Building HVAC-TES Control

Code and configurations accompanying the paper
**"Thermal-Load-Decoupled Reinforcement Learning for Cross-Building HVAC-TES Control."**

TLD-RL places the cross-building transfer interface at the **aggregate thermal load**.
A building-specific LSTM predictor compresses the heterogeneous zone-level state into a
fixed-dimensional load-forecast trajectory, and a single shared Soft Actor-Critic (SAC)
supervisory policy acts on that trajectory together with the HVAC-TES state, electricity
tariff, and weather. Adapting to a new building reduces to **refitting the local predictor
while the policy is reused frozen** (zero-shot), so one pre-trained policy transfers across
buildings of differing zone count.

The experiments build on the open-source [Sinergym](https://github.com/ugr-sail/sinergym)
framework (EnergyPlus + Gymnasium).

## Repository structure

| Path | Contents |
|------|----------|
| `sinergym/` | Sinergym core library and building models (`sinergym/data/buildings/`) |
| `studies/case_study/config/` | Building, method, recipe, and case definitions (YAML) |
| `studies/case_study/core/` | Registry, runner, environment builder, weight transfer, physics-based wrappers/rewards |
| `studies/case_study/methods/` | From-scratch baseline, padding-based transfer (full and frozen-feature fine-tuning, zero-shot), and the load-decoupled transfer |
| `studies/case_study/cli/` | Run, inspect, plot, predictor-training, and seed-aggregation entry points |
| `studies/case_study/scripts/` | Per-source-seed zero-shot driver and the held-out-weather re-evaluation |
| `studies/case_study/analysis/` | Paper-table and figure generation |
| `tests/` | Test suite |

## Buildings

| Tag | Building | Zones | Role |
|-----|----------|-------|------|
| `office_5zone` (OF) | 5-zone office | 5 | source and target |
| `office_lbuilding_8zone` (LB) | L-shaped office | 8 | source and target |
| `warehouse_3zone` (WH) | warehouse | 3 | source and target |
| `mediumoffice_15zone` (MO) | medium office | 15 | source and target |
| `restaurant_3zone` (RE) | full-service restaurant | 2 cooled | added target |
| `stripmall_10zone` (SM) | strip mall | 10 | added target |
| `retail_4zone` (RT) | stand-alone retail | 4 | added target |
| `fastfood_2zone` (QS) | quick-service restaurant | 2 | added target |

All buildings share the chilled-water TES plant, the `[5, 10]` C supervisory setpoint
range, the **SMUD time-of-use tariff**, and TMY3 Chicago O'Hare weather: 15-min
supervisory interval, 12-h look-ahead, cooling season June 1 - September 30,
seeds 42/123/777/2024/2025. The four added buildings are DOE prototype models whose
chiller and tank are sized by the paper's rules (chiller = season-peak RBC load / 0.9,
tank = 3.5 h at rated capacity); their physical parameters are in
`studies/case_study/config/buildings.yaml`.

## Installation

```bash
poetry install --no-interaction --with dev
```

Requires Python >= 3.12 and a working EnergyPlus installation (see the Sinergym documentation).

## Reproducing the case study

The pipeline is configuration-driven: every experiment is a case id resolved from
`studies/case_study/config/`. Groups collect the cases behind each part of the paper:

| Group | Cases | Contents |
|-------|-------|----------|
| `paper_4building` | 44 | four-building matrix: datasets, RBC, TLD-RL / E2E / padded sources, 12 zero-shot and 12 Padding fine-tuning pairs |
| `paper_extension` | 28 | added buildings: datasets, RBC, 16 zero-shot pairs from the four sources |
| `r1_padff` | 12 | Padding with frozen features (first hidden layer of actor and critics fixed) |
| `r1_ppo` | 8 | TLD-RL and Padding sources trained with PPO |
| `r1_ppo_zs_tld`, `r1_ppo_zs_pad` | 12 + 12 | PPO zero-shot transfer (Padding-PPO without fine-tuning) |
| `r1_comfort` | 16 | TLD-RL sources with the comfort-weighted reward (four weights) |

```bash
# list all experiment cases / expand a group
python -m studies.case_study.cli.run --list
python -m studies.case_study.cli.run --group paper_extension --dry-run

# 1) operation-log datasets (fixed-setpoint and incumbent-RBC logs); repeat
#    for every building
python -m studies.case_study.cli.run --case build_office_observation_dataset
python -m studies.case_study.cli.run --case build_office_observation_dataset_rbc

# 2) per-building LSTM load predictors (Stage 1; canonical fixed-setpoint log)
python -m studies.case_study.cli.train_predictor --building office_5zone --epochs 50

#    Stage-3 target predictors are refit on the incumbent-RBC log:
python -m studies.case_study.cli.train_predictor --building office_5zone --epochs 50 \
  --data artifacts/case_study/exports/build_office_observation_dataset_rbc/observations_processed.csv \
  --output artifacts/case_study/predictors/office_5zone_lstm_rbclog.pt

# 3) the four-building matrix (five seeds, as reported in the paper)
python -m studies.case_study.cli.run --group paper_4building --default-seeds

# 4) zero-shot deployment of every source seed (exports <case>__src<seed>/),
#    e.g. the added buildings or the PPO arms
python -m studies.case_study.scripts.run_r1_zs --group paper_extension \
  --seeds 42 123 777 2024 2025

# aggregate seeded runs into mean +- s.d.
python -m studies.case_study.cli.aggregate_seeds --case <case_id>
```

Held-out weather: every building registers weather variants of its environment, so a case
runs under TMYx Chicago 2007-2021, New York JFK, or Pittsburgh by replacing `-mixed-` in
its environment id with `-mixedx-`, `-nyc-`, or `-pit-`.
`studies/case_study/scripts/run_r1_e5.py` re-evaluates the source final models this way.

### Tables and figures

| Script (`studies/case_study/analysis/`) | Output |
|------------------------------------------|--------|
| `extract_paper_tables.py` | source-domain and transfer tables of the four-building matrix |
| `r1_padff.py` | Padding fine-tuning columns (full and frozen features) |
| `r1_ppo.py` | PPO replication |
| `r1_inclusion.py --candidate restaurant` (or `stripmall`, `retail`, `fastfood`) | inclusion checks and S* of an added building |
| `r1_x_matrix.py`, `fig_extension_matrix.py` | zero-shot matrix of the added buildings |
| `r1_sstar_sweep.py`, `fig_sstar.py` | controlled S* sweep |
| `r1_comfort_c2.py` | comfort-weighted reward |
| `plot_predictor.py`, `fig_operating_profile.py`, `fig_adoption_v7.py` | predictor, operating-profile and adoption figures |

Figures are written to `artifacts/case_study/figures/`, analysis outputs to
`artifacts/case_study/results/`.

The paper's data-center fleet extension is not part of this release.

## Citation

If you use this code, please cite the accompanying paper. A BibTeX entry will be added to
`CITATION.bib` upon publication.

## Acknowledgements

Built on [Sinergym](https://github.com/ugr-sail/sinergym); please also credit the original
framework and consult its repository for the underlying simulation engine and license.

## License

See `LICENSE`.
