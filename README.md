# From open to closed societies - agent based simulations
Replication repository for 
"From open to closed societies: Inequality, migration, and women's rights". Journal of Development Economics, 178, 103607. https://doi.org/10.1016/j.jdeveco.2025.103607

This repository contains the agent-based simulation (Python) together with R scripts that process the simulation output and produce the figures and tables reported in the paper.

There are two ways to use this repository:

1. **Reproduce the figures from the simulation data shipped with the repo.** All raw simulation outputs used in the paper are included under `datacloud/processed/`. The simulations themselves with 1000 iterations might take from several hours to days. However you can regenerate every figure and table without re-running the simulation using the pre-processed data. This only requires R. See [Reproducing the figures from the included data](#reproducing-the-figures-from-the-included-data).
   
2. **Re-run the agent-based simulation yourself.** You can regenerate the raw CSVs (or run new simulations with different parameters) using the Python model, and then feed the output into the same R pipeline. This requires Python. See [Running the simulation](#running-the-simulation).

## Repository structure

```
opentoclosesocieties/
├── README.md
├── requirements.txt
├── code/
│   ├── simulation/                
│   │   ├── simulate.py            
│   │   ├── smallsim_test.ipynb    
│   │   ├── commons/               
│   │   │   ├── simulation.py      # Simulation orchestration
│   │   │   ├── community.py       # Communities & their inheritance regime
│   │   │   ├── agent.py           # Individual agents
│   │   │   ├── couple.py          # Married couples & offspring
│   │   │   ├── family.py          # Family / household logic
│   │   │   ├── marriagemarket.py  # Matching market across communities
│   │   │   ├── matchers.py        # Matching algorithms
│   │   │   ├── topography.py      # Spatial structure between communities
│   │   │   ├── config.py          # Default model parameters
│   │   │   ├── helpers.py         # Utilities (timestamps, output columns)
│   │   │   └── datafile.py        # CSV writer for agent-level output
│   │   └── tests/                 # Unit tests
│   └── analysis/                  # R scripts for data prep & figures
│       ├── combine_clouddata.R    # Combines raw per-run CSVs into processed datasets
│       ├── fig1-share_patrilineal.R
│       ├── fig2-lockindomino.R
│       ├── fig3_egalitarian_unilinealassets.R
│       └── fig4_gini.R
├── datacloud/
│   ├── raw/                       # Per-simulation CSVs (one folder per simulation)
│   └── processed/                 # Combined CSVs consumed by the R scripts
└── figures/                       # Output figures and summary tables
```

## Requirements

### Python (simulation)

- Python 3.10+
- Packages listed in [requirements.txt](requirements.txt):
  - `numpy==1.26.4`
  - `joblib==1.3.2`
  - `matching==1.4.3`

Install with:

```bash
pip install -r requirements.txt
```

### R (analysis)

The analysis scripts use:

- `tidyverse`
- `here`
- `scales`

Install in R with:

```r
install.packages(c("tidyverse", "here", "scales"))
```

## Running the simulation

From [code/simulation](code/simulation):

```bash
cd code/simulation
python simulate.py
```

You will be prompted for a `filetag` (e.g. `baseline`, `domino1`, `lockin1`). Output CSVs are written to `data/raw/<filetag>/`.

Key parameters set in [code/simulation/simulate.py](code/simulation/simulate.py):

| Parameter | Default | Notes |
|---|---|---|
| `number_of_simulations` | 8 | 1000 in the paper |
| `number_of_generations` | 15 | |
| `number_of_cohorts` | 10 | |
| `num_coms` | 7 | Number of communities |
| `com_size` | 160 | Agents per community |
| `write_agent_data` | `False` | Set `True` to dump per-agent rows |

Simulations are dispatched in parallel via `joblib` (`n_jobs=-1`).

### Model parameters

Default model parameters live in [code/simulation/commons/config.py](code/simulation/commons/config.py) and can be overridden per-simulation via the `overrides` argument to `Simulation(...)`. They control utility weights, inheritance rules (`init_egalitarian`, `asset_inheritance_egalitarian`, `membership_inheritance`), the marriage market (`proposer`), spatial structure (`topography_structure`), and switching thresholds between egalitarian and patrilineal regimes (`switch_threshold_to_egal_*`, `switch_threshold_to_patri_*`, `switch_rule`, `switchable_communities`).

### Parameters for replicating the simulations in the paper

> **Terminology note.** The paper uses the term **unilineal** membership, while the code uses **patrilineal** (e.g. `switch_threshold_to_patri_*`, `switch_to_patri`, `is_patrilineal`). The two refer to the same regime in this model — the simulation only implements the patrilineal case as a representative unilineal system.

Each subfolder of `datacloud/raw/` (and the corresponding processed CSV in `datacloud/processed/`) corresponds to one of the simulations reported in the paper. They differ from the defaults in [config.py](code/simulation/commons/config.py) only in the parameters listed below; everything else uses the default values (`K = 7` communities of `n = 160` agents, 15 generations × 10 cohorts, 1000 replicates).

| simulation | Paper reference | `init_egalitarian` | `asset_inheritance_egalitarian` | `switchable_communities` | Description |
|---|---|---|---|---|---|
| `baseline` | Prediction 1  | `True` | `True` | `None` (all switchable) | Polycentric baseline: every community starts egalitarian and is free to transition to a unilineal membership system. |
| `assetpatri` | Prediction 4 | `True` | `False` | `None` (all switchable) | Same as `baseline`, but private assets are inherited only by sons (unilineal asset inheritance). Used to show that the inheritance regime for private assets does not drive aggregate migration or institutional change. |
| `domino1` … `domino7` | Prediction 3  | `True` | `True` | `[k]` for `k = 1,…,7` | Only the community of wealth rank `k` may switch; all other communities are fixed in egalitarian. Sterilises migratory pressure from neighbours so that the gap to the baseline isolates the domino effect. |
| `lockin1` … `lockin7` | Prediction 3  | `False` | `True` | `[k]` for `k = 1,…,7` | Community `k` starts egalitarian (the switchable community is always created egalitarian, regardless of `init_egalitarian`) while all other communities are fixed unilineal. Measures how strongly migratory pressure from a closed neighbourhood pushes the lone open community to also close (lock-in). |

All simulations use the default `switch_rule = "both"` (transitions in either direction are allowed for switchable communities). The `switchable_communities = [k]` mechanism in [code/simulation/commons/simulation.py](code/simulation/commons/simulation.py) creates community `k` as egalitarian and switchable, and forces all other communities to be non-switchable — set to patrilineal when `init_egalitarian = False` (lock-in) or kept egalitarian when `init_egalitarian = True` (no-domino). Community wealth ranks 1 (richest) to 7 (poorest) are assigned by the simulation based on initial commons endowments.

## Reproducing the figures from the included data

The repository ships with the full set of raw simulation outputs under `datacloud/raw/`, organized one folder per simulation (`baseline`, `assetpatri`, `domino1`–`domino7`, `lockin1`–`lockin7`). Each folder contains one CSV per simulation replicate.

**Generate the figures and tables.** Each script is self-contained and reads from `datacloud/processed/`:

   ```r
   source("code/analysis/fig1-share_patrilineal.R")           # Share of unilineal communities 
   source("code/analysis/fig2-lockindomino.R")                # Baseline comparison Domino & lock-in dynamics 
   source("code/analysis/fig3_egalitarian_unilinealassets.R") # Private assets comparison
   source("code/analysis/fig4_gini.R")                        # Inequality / Gini 
   ```

   Figures and summary tables are written to [figures/](figures/).



