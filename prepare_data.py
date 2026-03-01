"""
Download and prepare BBB permeability dataset from TDC.

Creates 1000 train + 100 test tasks from BBB_Martins dataset (~1,975 molecules).
Two task types:
  - Classification (700 train + 70 test): predict BBB+ or BBB-
  - Modification (300 train + 30 test): modify BBB- molecule to become BBB+

Also trains a Random Forest oracle on Morgan fingerprints for modification verification.

Run once locally: python prepare_data.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from tdc.single_pred import ADME

CLS_TRAIN = 820
CLS_TEST = 80
MOD_TRAIN = 180
MOD_TEST = 20

RANDOM_STATE = 42
FP_RADIUS = 2
FP_NBITS = 2048
TANIMOTO_MIN = 0.3


def canonicalize_smiles(smiles: str) -> str | None:
    """Canonicalize SMILES via RDKit. Returns None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def smiles_to_fp(smiles: str) -> np.ndarray | None:
    """Convert SMILES to Morgan fingerprint array."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=FP_NBITS)
    arr = np.zeros(FP_NBITS, dtype=np.int8)
    for i in range(FP_NBITS):
        arr[i] = fp[i]
    return arr


def smiles_to_bitvect(smiles: str):
    """Convert SMILES to RDKit fingerprint bit vector (for Tanimoto)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=FP_NBITS)


def make_cls_question(smiles: str) -> str:
    return (
        "You are a drug discovery expert specializing in blood-brain barrier (BBB) "
        "permeability prediction.\n\n"
        f"Given the molecule with SMILES notation: {smiles}\n\n"
        "Predict whether this molecule can penetrate the blood-brain barrier:\n"
        "- BBB+ (1): The molecule CAN cross the blood-brain barrier\n"
        "- BBB- (0): The molecule CANNOT cross the blood-brain barrier\n\n"
        "Submit your prediction as 0 or 1 using the submit_prediction tool."
    )


def make_mod_question(smiles: str) -> str:
    return (
        "You are a medicinal chemist specializing in CNS drug design and "
        "blood-brain barrier optimization.\n\n"
        f"Given the molecule with SMILES notation: {smiles}\n\n"
        "This molecule is classified as BBB- (cannot cross the blood-brain barrier).\n\n"
        "Your task: Modify this molecule to make it BBB+ (able to cross the "
        "blood-brain barrier) while maintaining structural similarity to the original.\n\n"
        "Requirements:\n"
        "- The modified molecule must be a valid SMILES string\n"
        "- It must be structurally similar to the original (Tanimoto similarity > 0.3)\n"
        "- It must not be identical to the original\n"
        "- An oracle classifier must confirm the modified molecule is BBB+\n\n"
        "Consider these BBB optimization strategies:\n"
        "- Increase lipophilicity (add methyl groups, reduce polar surface area)\n"
        "- Reduce hydrogen bond donors\n"
        "- Reduce molecular weight if too high\n"
        "- Add halogen substituents to increase membrane permeability\n\n"
        "Submit your modified SMILES using the submit_modification tool."
    )


def main():
    print("Downloading BBB_Martins dataset...")
    data = ADME(name="BBB_Martins")
    df = data.get_data()
    df = df.dropna(subset=["Drug", "Y"])
    df = df.drop_duplicates(subset=["Drug"])
    df["Y"] = df["Y"].astype(int)
    df = df[df["Y"].isin([0, 1])]

    # Canonicalize and validate SMILES
    print("Validating and canonicalizing SMILES...")
    df["canonical"] = df["Drug"].apply(canonicalize_smiles)
    df = df.dropna(subset=["canonical"])
    df = df.drop_duplicates(subset=["canonical"])
    df["Drug"] = df["canonical"]
    df = df.drop(columns=["canonical"])

    print(f"Total valid molecules: {len(df)}")
    print(f"Class distribution: {df['Y'].value_counts().to_dict()}")

    # Split into BBB+ and BBB- pools
    bbb_pos = df[df["Y"] == 1].sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    bbb_neg = df[df["Y"] == 0].sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"BBB+ pool: {len(bbb_pos)}, BBB- pool: {len(bbb_neg)}")

    # --- Train oracle model on full dataset ---
    print("\nTraining oracle model...")
    all_fps = []
    all_labels = []
    for _, row in df.iterrows():
        fp = smiles_to_fp(row["Drug"])
        if fp is not None:
            all_fps.append(fp)
            all_labels.append(row["Y"])

    X = np.array(all_fps)
    y = np.array(all_labels)

    oracle = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # Cross-validate
    cv_scores = cross_val_score(oracle, X, y, cv=5, scoring="roc_auc")
    print(f"Oracle 5-fold CV AUROC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Train on full dataset
    oracle.fit(X, y)
    train_acc = oracle.score(X, y)
    print(f"Oracle train accuracy: {train_acc:.4f}")

    # --- Precompute fingerprints for solvability check ---
    print("\nPrecomputing fingerprints for solvability filtering...")
    pos_fps = []
    pos_smiles_list = bbb_pos["Drug"].tolist()
    for s in pos_smiles_list:
        fp = smiles_to_bitvect(s)
        if fp is not None:
            pos_fps.append(fp)

    # --- Allocate modification tasks FIRST (they need BBB- molecules) ---
    # Filter: oracle must predict BBB- (0) AND must have a BBB+ neighbor with Tanimoto > 0.3
    mod_total = MOD_TRAIN + MOD_TEST  # 330

    print(f"\nFiltering modification candidates (need {mod_total})...")
    mod_candidates = []
    for idx, row in bbb_neg.iterrows():
        smiles = row["Drug"]

        # Check oracle predicts BBB- correctly
        fp_arr = smiles_to_fp(smiles)
        if fp_arr is None:
            continue
        oracle_pred = oracle.predict(fp_arr.reshape(1, -1))[0]
        if oracle_pred != 0:
            continue  # Oracle mispredicts this as BBB+, skip

        # Check there exists at least one BBB+ neighbor with Tanimoto > threshold
        fp_bv = smiles_to_bitvect(smiles)
        if fp_bv is None:
            continue
        has_neighbor = False
        for ref_fp in pos_fps:
            sim = DataStructs.TanimotoSimilarity(fp_bv, ref_fp)
            if sim > TANIMOTO_MIN:
                has_neighbor = True
                break
        if not has_neighbor:
            continue

        mod_candidates.append(idx)

    print(f"Modification candidates passing filters: {len(mod_candidates)}")

    if len(mod_candidates) < mod_total:
        print(f"Warning: only {len(mod_candidates)} candidates, reducing mod tasks")
        mod_total = len(mod_candidates)

    # Select modification molecules
    mod_indices = mod_candidates[:mod_total]
    mod_df = bbb_neg.loc[mod_indices].reset_index(drop=True)
    mod_smiles_set = set(mod_df["Drug"].tolist())

    print(f"Selected {len(mod_df)} modification tasks")

    # --- Allocate classification tasks from remaining molecules ---
    cls_total = CLS_TRAIN + CLS_TEST  # 770

    # Use BBB- molecules NOT used for modification
    cls_neg = bbb_neg[~bbb_neg["Drug"].isin(mod_smiles_set)].reset_index(drop=True)

    # Target balanced classes: ~50% BBB+ and ~50% BBB-
    n_cls_neg = min(len(cls_neg), cls_total // 2)
    n_cls_pos = cls_total - n_cls_neg

    cls_pos_df = bbb_pos.head(n_cls_pos).reset_index(drop=True)
    cls_neg_df = cls_neg.head(n_cls_neg).reset_index(drop=True)

    # Stratified test/train split: maintain class ratio in both splits
    cls_test_ratio = CLS_TEST / cls_total
    n_test_pos = round(n_cls_pos * cls_test_ratio)
    n_test_neg = CLS_TEST - n_test_pos
    n_train_pos = n_cls_pos - n_test_pos
    n_train_neg = n_cls_neg - n_test_neg

    cls_test_df = pd.concat([
        cls_pos_df.head(n_test_pos),
        cls_neg_df.head(n_test_neg),
    ]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    cls_train_df = pd.concat([
        cls_pos_df.iloc[n_test_pos:],
        cls_neg_df.iloc[n_test_neg:],
    ]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"\nClassification tasks: {n_cls_pos + n_cls_neg} (BBB+: {n_cls_pos}, BBB-: {n_cls_neg})")
    print(f"  Train: {len(cls_train_df)} (BBB+: {n_train_pos}, BBB-: {n_train_neg})")
    print(f"  Test:  {len(cls_test_df)} (BBB+: {n_test_pos}, BBB-: {n_test_neg})")

    # --- Build task lists ---
    tasks = []

    # Classification tasks - test
    for idx, row in cls_test_df.iterrows():
        task = {
            "task_id": f"bbb_cls_test_{idx}",
            "task_type": "classification",
            "smiles": row["Drug"],
            "question": make_cls_question(row["Drug"]),
            "answer": int(row["Y"]),
        }
        tasks.append(task)

    # Classification tasks - train
    for idx, row in cls_train_df.iterrows():
        task = {
            "task_id": f"bbb_cls_train_{idx}",
            "task_type": "classification",
            "smiles": row["Drug"],
            "question": make_cls_question(row["Drug"]),
            "answer": int(row["Y"]),
        }
        tasks.append(task)

    # Modification tasks - stratified by index
    mod_test_df = mod_df.head(MOD_TEST)
    mod_train_df = mod_df.iloc[MOD_TEST:]

    for idx, row in mod_test_df.iterrows():
        task = {
            "task_id": f"bbb_mod_test_{idx}",
            "task_type": "modification",
            "smiles": row["Drug"],
            "original_label": 0,
            "target_label": 1,
            "question": make_mod_question(row["Drug"]),
        }
        tasks.append(task)

    for idx, row in mod_train_df.iterrows():
        task = {
            "task_id": f"bbb_mod_train_{idx}",
            "task_type": "modification",
            "smiles": row["Drug"],
            "original_label": 0,
            "target_label": 1,
            "question": make_mod_question(row["Drug"]),
        }
        tasks.append(task)

    test_tasks = [t for t in tasks if "test" in t["task_id"]]
    train_tasks = [t for t in tasks if "train" in t["task_id"]]

    # Verify counts
    cls_train_count = sum(1 for t in train_tasks if t["task_type"] == "classification")
    cls_test_count = sum(1 for t in test_tasks if t["task_type"] == "classification")
    mod_train_count = sum(1 for t in train_tasks if t["task_type"] == "modification")
    mod_test_count = sum(1 for t in test_tasks if t["task_type"] == "modification")

    print(f"\nTrain tasks: {len(train_tasks)} (cls: {cls_train_count}, mod: {mod_train_count})")
    print(f"Test tasks: {len(test_tasks)} (cls: {cls_test_count}, mod: {mod_test_count})")

    # Verify class balance in classification
    cls_answers = [t["answer"] for t in tasks if t["task_type"] == "classification"]
    n_pos = sum(cls_answers)
    n_neg = len(cls_answers) - n_pos
    print(f"\nClassification answer balance: BBB+={n_pos} ({n_pos/len(cls_answers)*100:.1f}%), "
          f"BBB-={n_neg} ({n_neg/len(cls_answers)*100:.1f}%)")

    # Verify no SMILES overlap between train and test
    train_smiles = set(t["smiles"] for t in train_tasks)
    test_smiles = set(t["smiles"] for t in test_tasks)
    overlap = train_smiles & test_smiles
    print(f"Train/test SMILES overlap: {len(overlap)}")

    # --- Save outputs ---
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    with open(data_dir / "test.json", "w") as f:
        json.dump(test_tasks, f, indent=2)
    with open(data_dir / "train.json", "w") as f:
        json.dump(train_tasks, f, indent=2)

    joblib.dump(oracle, data_dir / "bbb_oracle.pkl")

    print(f"\nSaved {len(train_tasks)} train tasks and {len(test_tasks)} test tasks")
    print(f"Saved oracle model to {data_dir / 'bbb_oracle.pkl'}")


if __name__ == "__main__":
    main()
