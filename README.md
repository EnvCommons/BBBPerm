# BBBPerm

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/BBBPerm)

## Description

**BBBPerm** is an environment for evaluating agents on blood-brain barrier (BBB) permeability tasks. It includes two task types: classification (predicting whether a molecule can cross the BBB) and modification (proposing structural changes to a BBB-impermeable molecule to make it permeable). The dataset is derived from the [TDC BBB_Martins dataset](https://tdcommons.ai/single_pred_tasks/adme/#bbb-blood-brain-barrier-martins-et-al).

## Capabilities

- Predicting blood-brain barrier permeability from molecular SMILES notation
- Molecular property classification (BBB+ or BBB-)
- Molecular structure modification with structural similarity constraints
- Understanding structure-activity relationships for BBB permeability

## Compute Requirements

BBBPerm does not require a sandbox. It has minimal compute requirements.

## License

[CC BY 4.0](https://opensource.org/license/mit](https://creativecommons.org/licenses/by/4.0/).

## Tasks

There are two splits: train (1,000 tasks) and test (100 tasks). Tasks are derived from the TDC BBB_Martins dataset (~1,975 molecules) and include two task types:

- **Classification** (820 train / 80 test): Given a molecule's SMILES string, predict whether it is BBB+ (can cross, label 1) or BBB- (cannot cross, label 0). The dataset is approximately 70% BBB+ / 30% BBB-.
- **Modification** (180 train / 20 test): Given a BBB-impermeable molecule (BBB-), propose a structurally similar molecule that is BBB-permeable (BBB+). The modified molecule must pass a 6-step validation pipeline: valid SMILES, sanitization, single fragment, not identical to the original, Tanimoto similarity >= 0.3, and oracle model confirmation.

## Reward Structure

This is a sparse, verifiable reward environment. Each task has one graded submission.

- **Classification**: Binary reward. **1.0** for a correct prediction, **0.0** for incorrect.
- **Modification**: Binary reward. **1.0** if the modified molecule passes all six validation steps. **0.0** otherwise.
- **Ungraded submissions**: calling the tool for the other task type, a prediction other than 0 or 1, or a modified SMILES that fails to parse or sanitize, has several fragments, or is identical to the original returns **0.0** without grading. The episode stays open so the agent can resubmit.

We do not use LLM graders for this task.

## Data

Task data is derived from the [TDC BBB_Martins dataset](https://tdcommons.ai/single_pred_tasks/adme/#bbb-blood-brain-barrier-martins-et-al) (~1,975 molecules with binary BBB permeability labels). An oracle model (Random Forest on Morgan fingerprints, 2048-bit, radius 2, 5-fold CV AUROC ~0.87) is used for verifying modification tasks. Data files are stored on the OpenReward platform.

## Tools

Agents are given two environment-specific tools (one per task type):

- `submit_prediction`: Submit a BBB permeability classification (0 = BBB-, 1 = BBB+). Used for classification tasks.
- `submit_modification`: Submit a modified SMILES string with changed BBB permeability. Used for modification tasks. The molecule is validated through a 6-step pipeline.

## Time Horizon

BBBPerm is a single-turn environment. The agent receives a question and submits one answer. Each task requires one tool call, plus a resubmission after an ungraded submission.

[Statistics on average tool calls here]

## Environment Difficulty

[Statistics on environment difficulty here]

## Other Environment Requirements

There are no further environment requirements; BBBPerm works out of the box with the OpenReward endpoint without any secrets.

## Safety

Agents in BBBPerm are asked to predict or modify molecular properties related to blood-brain barrier permeability. The environment does not present direct safety risks, as agents only provide predictions or molecular modifications evaluated by an oracle model, with no access to external systems.

## Citations


```bibtex
@dataset{GRBBBPerm,
  author    = {General Reasoning Inc. Team},
  title     = {BBBPerm},
  year      = {2026},
  publisher = {OpenReward},
  url       = {https://openreward.ai/GeneralReasoning/BBBPerm}
}
```

```bibtex
@article{martins2012bayesian,
  title={A Bayesian approach to in silico blood-brain barrier penetration modeling},
  author={Martins, Ines Filipa and Teixeira, Ana L and Pinheiro, Luis and Falcao, Andre O},
  journal={Journal of Chemical Information and Modeling},
  volume={52},
  number={6},
  pages={1686--1697},
  year={2012},
  publisher={ACS Publications}
}
```
