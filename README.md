# BBBPerm - Blood-Brain Barrier Permeability Environment

An OpenReward environment for evaluating agents on blood-brain barrier (BBB) permeability tasks. Agents must either classify molecules by their ability to cross the BBB, or modify non-permeable molecules to become permeable.

## Task Types

### Classification (820 train / 80 test)

Given a molecule's SMILES, predict whether it can cross the blood-brain barrier:
- **BBB+ (1)**: Molecule can cross the BBB
- **BBB- (0)**: Molecule cannot cross the BBB

Grading: exact match. Reward is 1.0 for correct, 0.0 for incorrect.

### Modification (180 train / 20 test)

Given a BBB- molecule, modify it to become BBB+. The submitted molecule must pass all of:
1. Valid SMILES (RDKit parse + sanitize)
2. Single fragment (no salts/mixtures)
3. Not identical to the original
4. Tanimoto similarity > 0.3 to the original (Morgan fingerprints, radius 2)
5. Oracle Random Forest classifier confirms BBB+

Grading: binary. Reward is 1.0 if all checks pass, 0.0 otherwise.

## Data

- **Source**: [TDC BBB_Martins](https://tdcommons.ai/single_pred_tasks/adme/) (~1,975 molecules, Martins et al. 2012)
- **Splits**: 1,000 train + 100 test
- **Classification balance**: ~70% BBB+ / 30% BBB- (stratified across train and test)
- **Modification solvability**: All modification tasks are verified to have at least one structurally similar BBB+ molecule (Tanimoto > 0.3) and the oracle correctly classifies the source as BBB-
- **Oracle**: Random Forest on Morgan fingerprints (2048-bit, radius 2), 5-fold CV AUROC ~0.87

## Tools

| Tool | Task Type | Input | Description |
|------|-----------|-------|-------------|
| `submit_prediction` | Classification | `prediction: int` (0 or 1) | Submit BBB permeability class |
| `submit_modification` | Modification | `modified_smiles: str` | Submit modified SMILES string |

## Setup

### Generate data (run once)

```bash
pip install PyTDC pandas rdkit-pypi scikit-learn joblib
python prepare_data.py
```

This downloads the BBB_Martins dataset, trains the oracle model, and writes `data/train.json`, `data/test.json`, and `data/bbb_oracle.pkl`.

### Run locally

```bash
pip install -r requirements.txt
python server.py
```

### Run with Docker

```bash
docker build -t bbbperm .
docker run -p 8080:8080 bbbperm
```

### Test with an agent

```bash
export OPENAI_API_KEY=your_key
python test_agent.py
```

## File Structure

```
bbbperm/
├── bbbperm.py          # Environment class (classification + modification tools)
├── server.py           # Minimal server wrapper
├── test_agent.py       # Agent integration test
├── prepare_data.py     # Data download and oracle training
├── requirements.txt    # Runtime dependencies
├── Dockerfile
└── data/
    ├── train.json      # 1,000 training tasks
    ├── test.json       # 100 test tasks
    └── bbb_oracle.pkl  # Random Forest BBB classifier
```
