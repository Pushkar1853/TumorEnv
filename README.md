# Tumor Micro-environment calibration (AgentTorch)

## Agents: 
* Tumor cell agent
* Immune cell agent

## Installation (currently)

1. `git clone -b substeps https://github.com/AgentTorch/TumorEnv.git`
2. `python3 -m venv agent-torch-env`
3. `source agent-torch-env/bin/activate`
4. `pip install -r requirements.txt` 
5. `pip install agent-torch==0.2.4`
6. copy helpers>general.py file to site-packages>AgentTorch>helpers>general.py   
7. copy the controller.py file to site-packages>AgentTorch>controller.py
8. `python3 trainer.py -c config.yaml`

## Purpose

* We develop methods to calibrate clinical Agent-Based Models (ABMs) directly from biopsies
to test under the Spatial Agreement Measure (SAM) Metric, minimizing the number of biopsy samples
taken.
* We design a novel multi-modal calibrated ABM pipeline to apply gradient-based ABMs to
simulate tumor-immune cell interactions. (for Cytotoxic CD8+ T Cells in multiple carcinomas and
melanoma cases)

## Pipeline

* The images are converted to tumors and immune location matrices via the [Deepliif model](https://deepliif.org/).

* These locations are fed to the Agent-Based Model as initial location matrices.

* The substeps of the Agent-Based Model(ABM) are in the following order:
  1. Tumor cell proliferation
  2. Tumor cell migration
  3. Immune cell proliferation
  4. Immune cell migration
  5. Tumor and Immune cell Interaction
  6. Tumor cell Death
  7. Immune cell Death

