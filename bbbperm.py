"""
BBBPerm - Blood-Brain Barrier Permeability Environment

Single-turn environment with two task types:
  - Classification: predict BBB+ (1) or BBB- (0) for a molecule
  - Modification: modify a BBB- molecule to become BBB+, verified by oracle

Data source: TDC BBB_Martins dataset (~1,975 molecules).
Classification: binary reward (exact match).
Modification: binary reward (valid SMILES + structural similarity + oracle confirmation).
"""

import json
import os
from pathlib import Path
from typing import List

import joblib
import numpy as np
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from openreward.environments import (
    Environment,
    JSONObject,
    Split,
    TextBlock,
    ToolOutput,
    tool,
)

if os.path.exists("/orwd_data"):
    ENV_PATH = Path("/orwd_data")
else:
    ENV_PATH = Path(__file__).parent

TANIMOTO_MIN = 0.3
FP_RADIUS = 2
FP_NBITS = 2048


def load_all_tasks() -> dict[str, list[dict]]:
    data_dir = ENV_PATH / "data"
    all_tasks = {}
    for split in ["train", "test"]:
        json_file = data_dir / f"{split}.json"
        if json_file.exists():
            with open(json_file, "r", encoding="utf-8") as f:
                all_tasks[split] = json.load(f)
        else:
            print(f"Warning: {json_file} not found")
            all_tasks[split] = []
    return all_tasks


ALL_TASKS = load_all_tasks()

ANSWERS = {}
for _split_tasks in ALL_TASKS.values():
    for _task in _split_tasks:
        if _task["task_type"] == "classification":
            ANSWERS[_task["task_id"]] = {"value": _task["answer"]}
        else:
            ANSWERS[_task["task_id"]] = {
                "original_smiles": _task["smiles"],
                "original_label": _task["original_label"],
                "target_label": _task["target_label"],
            }

print(f"Loaded {len(ANSWERS)} BBBPerm tasks")

# Load oracle model for modification verification
ORACLE_MODEL = None
_oracle_path = ENV_PATH / "data" / "bbb_oracle.pkl"
if _oracle_path.exists():
    ORACLE_MODEL = joblib.load(_oracle_path)
    print("Loaded BBB oracle model")
else:
    print(f"Warning: Oracle model not found at {_oracle_path}")


# Reward for a submission made after the task has already been graded. Negative
# so repeat submissions are actively discouraged, not merely left unscored.
REPEAT_SUBMISSION_PENALTY = -0.1


class BBBPermTaskSpec(BaseModel):
    task_id: str
    task_type: str
    smiles: str
    question: str
    original_label: int | None = None
    target_label: int | None = None


class SubmitClassificationInput(BaseModel):
    prediction: int = Field(
        ..., description="Your predicted class: 0 (BBB-, cannot cross) or 1 (BBB+, can cross)"
    )


class SubmitModificationInput(BaseModel):
    modified_smiles: str = Field(
        ..., description="Modified SMILES string that should have changed BBB permeability"
    )


def _smiles_to_fp_array(smiles: str) -> np.ndarray | None:
    """Convert SMILES to Morgan fingerprint numpy array."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=FP_NBITS)
    arr = np.zeros((1, FP_NBITS))
    for i in range(FP_NBITS):
        arr[0, i] = fp[i]
    return arr


class BBBPerm(Environment):
    """
    Blood-brain barrier permeability environment.

    Two task types:
      - Classification: predict BBB+ (1) or BBB- (0). Binary reward.
      - Modification: modify BBB- molecule to BBB+. Oracle-verified binary reward.
    """

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.validated = BBBPermTaskSpec.model_validate(task_spec)

        if self.validated.task_id not in ANSWERS:
            raise ValueError(f"Task {self.validated.task_id} not found in ANSWERS")

        self.answer = ANSWERS[self.validated.task_id]

        # Graded submissions this session, shared by both submit tools -- a task is
        # either a classification or a modification, never both, so one episode gets
        # one graded attempt either way. Only the first is rewarded: a wrong binary
        # classification implies the true label, so a second guess would always be
        # right; the modification feedback says exactly why the oracle rejected a
        # molecule, which is a search oracle over the target label. Ungraded
        # submissions (wrong tool, invalid input) are not counted and do not end the
        # episode.
        self.submitted = 0

    @classmethod
    def list_splits(cls) -> list[Split]:
        return [
            Split(name="train", type="train"),
            Split(name="test", type="test"),
        ]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split not in ALL_TASKS:
            return []
        return [
            {k: v for k, v in task.items() if k != "answer"}
            for task in ALL_TASKS[split]
        ]

    async def get_prompt(self) -> List[TextBlock]:
        return [TextBlock(text=self.validated.question)]

    @tool
    async def submit_prediction(self, params: SubmitClassificationInput) -> ToolOutput:
        """Submit your BBB permeability classification (0 = BBB-, 1 = BBB+)."""
        if self.submitted > 0:
            return ToolOutput(
                blocks=[TextBlock(text="A prediction has already been submitted for this task. "
                                       "This episode is over: it is not re-graded, and repeat "
                                       "submissions are penalised (reward -0.1).")],
                metadata={"already_submitted": True, "submission_count": self.submitted},
                reward=REPEAT_SUBMISSION_PENALTY,
                finished=True,
            )

        if self.validated.task_type != "classification":
            return ToolOutput(
                blocks=[TextBlock(text="Error: This task requires molecule modification, not classification. "
                                       "Nothing was graded; use submit_modification.")],
                metadata={"error": "wrong_tool"},
                reward=0.0,
                finished=False,
            )

        predicted = params.prediction
        if predicted not in (0, 1):
            return ToolOutput(
                blocks=[TextBlock(text=f"Error: Prediction must be 0 (BBB-) or 1 (BBB+), got {predicted}. "
                                       "Nothing was graded; resubmit with 0 or 1.")],
                metadata={"error": "invalid_prediction", "predicted": predicted},
                reward=0.0,
                finished=False,
            )

        actual = self.answer["value"]
        correct = predicted == actual
        reward = 1.0 if correct else 0.0

        label_map = {0: "BBB-", 1: "BBB+"}
        desc_map = {0: "impermeable", 1: "permeable"}

        if correct:
            feedback = (
                f"Correct! The molecule is {label_map[actual]} "
                f"(blood-brain barrier {desc_map[actual]}).\n"
                f"Reward: {reward:.1f}"
            )
        else:
            feedback = (
                f"Incorrect. You predicted {label_map[predicted]}.\n"
                f"Reward: {reward:.1f}"
            )

        self.submitted += 1

        return ToolOutput(
            blocks=[TextBlock(text=feedback)],
            metadata={
                "task_id": self.validated.task_id,
                "smiles": self.validated.smiles,
                "predicted": predicted,
                "correct": correct,
            },
            reward=reward,
            finished=True,
        )

    @tool
    async def submit_modification(self, params: SubmitModificationInput) -> ToolOutput:
        """Submit a modified molecule with changed BBB permeability. The molecule must be valid, structurally similar to the original, and pass oracle verification."""
        if self.submitted > 0:
            return ToolOutput(
                blocks=[TextBlock(text="A modification has already been submitted for this task. "
                                       "This episode is over: it is not re-graded, and repeat "
                                       "submissions are penalised (reward -0.1).")],
                metadata={"already_submitted": True, "submission_count": self.submitted},
                reward=REPEAT_SUBMISSION_PENALTY,
                finished=True,
            )

        if self.validated.task_type != "modification":
            return ToolOutput(
                blocks=[TextBlock(text="Error: This task requires classification, not modification. "
                                       "Nothing was graded; use submit_prediction.")],
                metadata={"error": "wrong_tool"},
                reward=0.0,
                finished=False,
            )

        submitted = params.modified_smiles.strip()
        original_smiles = self.answer["original_smiles"]
        target_label = self.answer["target_label"]

        # Step 1: Parse SMILES. An empty string parses to a molecule with no atoms,
        # which is not a modification either.
        mol = Chem.MolFromSmiles(submitted)
        if mol is None or mol.GetNumAtoms() == 0:
            return self._mod_failure("Invalid SMILES - could not parse.", submitted, graded=False)

        # Step 2: Sanitize
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            return self._mod_failure(f"SMILES sanitization failed: {e}", submitted, graded=False)

        # Step 3: Reject multi-fragment molecules (e.g. salts)
        frags = Chem.GetMolFrags(mol)
        if len(frags) > 1:
            return self._mod_failure(
                "Multi-fragment SMILES are not accepted. Submit a single molecule.",
                submitted,
                graded=False,
            )

        # Step 4: Not identical to original
        canonical_submitted = Chem.MolToSmiles(mol, canonical=True)
        original_mol = Chem.MolFromSmiles(original_smiles)
        canonical_original = Chem.MolToSmiles(original_mol, canonical=True)
        if canonical_submitted == canonical_original:
            return self._mod_failure("Modified molecule is identical to the original.", submitted, graded=False)

        # Step 5: Tanimoto similarity check
        fp_orig = AllChem.GetMorganFingerprintAsBitVect(original_mol, radius=FP_RADIUS, nBits=FP_NBITS)
        fp_mod = AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=FP_NBITS)
        tanimoto = DataStructs.TanimotoSimilarity(fp_orig, fp_mod)
        if tanimoto < TANIMOTO_MIN:
            return self._mod_failure(
                f"Tanimoto similarity {tanimoto:.3f} is below threshold {TANIMOTO_MIN}. "
                f"The modification must be structurally related to the original.",
                submitted,
            )

        # Step 6: Oracle prediction
        # A missing oracle is an infra fault, not a verdict on the molecule: raise so
        # the call fails without a reward and without ending the episode.
        if ORACLE_MODEL is None:
            raise RuntimeError("BBB oracle model not loaded")

        fp_array = _smiles_to_fp_array(canonical_submitted)
        if fp_array is None:
            return self._mod_failure("Could not compute fingerprint for modified molecule.", submitted, graded=False)

        oracle_pred = int(ORACLE_MODEL.predict(fp_array)[0])

        if oracle_pred != target_label:
            label_map = {0: "BBB-", 1: "BBB+"}
            return self._mod_failure(
                f"Oracle predicts {label_map[oracle_pred]}, "
                f"but target was {label_map[target_label]}.",
                submitted,
            )

        # All checks pass
        label_map = {0: "BBB-", 1: "BBB+"}
        feedback = (
            f"Modification accepted!\n\n"
            f"Original: {original_smiles} ({label_map[self.answer['original_label']]})\n"
            f"Modified: {canonical_submitted} (oracle confirms {label_map[target_label]})\n"
            f"Tanimoto similarity: {tanimoto:.3f}\n\n"
            f"Reward: 1.0"
        )
        self.submitted += 1

        return ToolOutput(
            blocks=[TextBlock(text=feedback)],
            metadata={
                "task_id": self.validated.task_id,
                "original_smiles": original_smiles,
                "modified_smiles": canonical_submitted,
                "tanimoto": round(tanimoto, 4),
                "oracle_prediction": oracle_pred,
                "target_label": target_label,
            },
            reward=1.0,
            finished=True,
        )

    def _mod_failure(self, reason: str, submitted: str, graded: bool = True) -> ToolOutput:
        """graded=False for input-validity rejections, which never reached the
        oracle and so neither consume the episode's one attempt nor end it."""
        if graded:
            self.submitted += 1
        feedback = (
            f"Modification rejected.\n\n"
            f"Reason: {reason}\n"
            f"Your submission: {submitted}\n\n"
            f"Reward: 0.0"
        )
        if not graded:
            feedback += "\nThis submission was not graded; submit a corrected molecule."
        return ToolOutput(
            blocks=[TextBlock(text=feedback)],
            metadata={
                "task_id": self.validated.task_id,
                "submitted": submitted,
                "valid": False,
                "reason": reason,
            },
            reward=0.0,
            finished=graded,
        )
