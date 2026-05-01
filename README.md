# From Open to Closed Institutions: Endogenous and Centralized Change in Commons Governance - Simlation Replication Materials
(Casari, Lisciandra, Saral)

## Description
This repository contains the replication materials for the simulation part of the paper "From Open to Closed Institutions: Endogenous and Centralized Change in Commons Governance". 

It contains the following parts:
 - Simulation code in Python
 - Data files for the simulations
 - Replication of the results presented in the paper

 To replicate the results, you can use the already generated data files provided in this repository. Alternatively, you can run the simulation code to generate the data files yourself (which might take a while depending on your machine).

 ## Replication with the data files (R)
To replicate the results using the data files we provide, run the R scripts in `code/analysis/`. Each script corresponds to a given plot or analysis in the paper. Note that the working directory should be set to the root of the repository. For convenience, you can use `run_all_analysis.R` to run all the analyses at once.

## Replication by running the simulations from scratch (Python + R)

1 - Install required packages:
Make sure you have Python 3.8 or higher installed. You can use a virtual environment to avoid conflicts with other packages.

```bash
pip install -r requirements.txt
```

2- Run the simulation code:
```bash
python code/simulation/simulate.py
```
You need to enter a name for the simulation run when prompted. This will create a folder in `simdata/raw/` with the simulation data. To match with the analysis scripts, you can use the names we use like `baseline` and others (see below for the different configurations). 

By default, the simulation will run with the baseline parameters (with 16 iterations instead of 1000 for testing purposes. Running it with 1000 iterations might take from several hours to several days depending on your configuration). You can change the parameters in `code/simulation/commons/config.py` to run with different parameters. (See the section below for the different configurations we used for the paper.)

3- Combine the simulation data:
```bash
Rscript code/simulation/combine_simulation_data.R
```

4- Now you ran run the R scripts in `code/analysis/` to replicate the results presented in the paper.

## Simulation Configurations (set them in `code/simulation/commons/config.py`):

## Baseline parameters (`baseline`)
```
"init_egalitarian": True,
"asset_inheritance_egalitarian": True,
"switch_rule": "both"
"switchable_communities": None, 
```

## Unilineal Assets (`assetpatri`)
```
"init_egalitarian": True,
"asset_inheritance_egalitarian": False,
"switch_rule": "both"
"switchable_communities": None, 
```

## Lock-in Effect (`lockin`)
You need to run this simulation for each 7 communities separately, as the switchable communities are defined by rank.
```
"init_egalitarian": False,
"asset_inheritance_egalitarian": True,
"switch_rule": "both",
"switchable_communities": [1], # set to 2, then 3, then 4, etc.
```

## Without the Domino Effect (`domino`)
You need to run this simulation for each 7 communities separately, as the switchable communities are defined by rank.
```
"init_egalitarian": True,
"asset_inheritance_egalitarian": True,
"switch_rule": "both",
"switchable_communities": [1], # set to 2, then 3, then 4, etc.
```
