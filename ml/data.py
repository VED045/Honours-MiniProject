from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

REQUIRED = ["train_transaction.csv", "train_identity.csv"]
ALL_FILES = REQUIRED + ["test_transaction.csv", "test_identity.csv", "sample_submission.csv"]
KNOWN_CATEGORIES = {"ProductCD", "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo"}
KNOWN_CATEGORIES.update(f"M{i}" for i in range(1, 10))
KNOWN_CATEGORIES.update(f"id_{i:02}" for i in range(12, 39))

def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def verify(root):
    root = Path(root)
    records = {}
    for name in REQUIRED:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Authenticate Kaggle and run scripts/download_dataset.py")
        cols = pd.read_csv(path, nrows=0).columns.tolist()
        required = {"TransactionID", "isFraud", "TransactionDT", "TransactionAmt"} if name.startswith("train_transaction") else {"TransactionID"}
        if not required.issubset(cols):
            raise ValueError(f"Required columns absent in {name}: {required - set(cols)}")
        seen = set()
        rows = 0
        for chunk in pd.read_csv(path, usecols=["TransactionID"] + (["isFraud", "TransactionDT"] if "isFraud" in cols else []), chunksize=100000):
            ids = chunk.TransactionID
            if ids.isna().any() or ids.duplicated().any() or seen.intersection(ids.tolist()):
                raise ValueError(f"Missing or duplicate join keys in {name}")
            seen.update(ids.tolist())
            if "isFraud" in chunk:
                if chunk.isFraud.isna().any() or not set(chunk.isFraud.unique()).issubset({0, 1}):
                    raise ValueError("Target must be binary and nonmissing")
                if chunk.TransactionDT.isna().any():
                    raise ValueError("Relative transaction time is missing")
            rows += len(chunk)
        records[name] = {"rows": rows, "columns": len(cols), "bytes": path.stat().st_size, "sha256": file_sha(path)}
    transaction_ids = set(pd.read_csv(root / REQUIRED[0], usecols=["TransactionID"]).TransactionID)
    identity_ids = set(pd.read_csv(root / REQUIRED[1], usecols=["TransactionID"]).TransactionID)
    if not identity_ids.issubset(transaction_ids):
        raise ValueError("Identity contains keys absent from transactions")
    records["optional_files"] = {n: {"exists": (root/n).is_file(), "bytes": (root/n).stat().st_size if (root/n).is_file() else 0} for n in ALL_FILES[2:]}
    records["fingerprint"] = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return records

def safe_join(transaction, identity):
    if transaction.TransactionID.isna().any() or transaction.TransactionID.duplicated().any():
        raise ValueError("Invalid transaction keys")
    if identity.TransactionID.isna().any() or identity.TransactionID.duplicated().any():
        raise ValueError("Invalid identity keys")
    result = transaction.merge(identity, on="TransactionID", how="left", validate="one_to_one")
    assert len(result) == len(transaction)
    return result

def synthetic():
    x, y = make_classification(n_samples=2400, n_features=10, n_informative=6, n_redundant=2, weights=[0.965, 0.035], flip_y=0.005, class_sep=1.3, random_state=120)
    frame = pd.DataFrame(x, columns=[f"V{i}" for i in range(1, 11)])
    frame["TransactionID"] = np.arange(len(frame))
    frame["TransactionDT"] = np.arange(len(frame)) * 120
    frame["TransactionAmt"] = np.exp(frame.V1.clip(-4, 4)) * 50
    frame["ProductCD"] = np.where(frame.V2 > 0, "W", "C")
    frame.loc[frame.index[::13], "V3"] = np.nan
    frame["isFraud"] = y
    return frame

def load_dataset(home, config):
    if config.dataset == "synthetic":
        frame = synthetic()
        fingerprint = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()
    else:
        root = Path(home) / "data/raw/ieee-cis"
        manifest = verify(root)
        frame = safe_join(pd.read_csv(root / REQUIRED[0], low_memory=False), pd.read_csv(root / REQUIRED[1], low_memory=False))
        fingerprint = manifest["fingerprint"]
    if config.expected_fingerprint and fingerprint != config.expected_fingerprint:
        raise ValueError("Dataset fingerprint changed; select the validated version again")
    # Label-free duplicate removal happens before splitting. Transaction identifiers/time are excluded.
    duplicated = frame.drop(columns=["TransactionID", "TransactionDT", "isFraud"]).duplicated(keep="first")
    duplicate_count = int(duplicated.sum())
    frame = frame.loc[~duplicated].copy()
    sampled = bool(config.max_rows and len(frame) > config.max_rows)
    if sampled:
        frame = frame.sample(config.max_rows, random_state=config.split_seed)
    return frame.reset_index(drop=True), {"fingerprint": fingerprint, "source": config.dataset, "demonstration": config.dataset == "synthetic", "development_sample": sampled, "duplicates_removed": duplicate_count}

def split_data(frame, config):
    if config.split == "temporal":
        ordered = frame.sort_values(["TransactionDT", "TransactionID"]).index.to_numpy()
        times = frame.loc[ordered, "TransactionDT"].to_numpy()
        cut1 = times[int(len(times) * (1 - config.validation_fraction - config.test_fraction))]
        cut2 = times[int(len(times) * (1 - config.test_fraction))]
        parts = [ordered[times < cut1], ordered[(times >= cut1) & (times < cut2)], ordered[times >= cut2]]
    else:
        trainval, test = train_test_split(frame.index.to_numpy(), test_size=config.test_fraction, stratify=frame.isFraud, random_state=config.split_seed)
        train, val = train_test_split(trainval, test_size=config.validation_fraction / (1 - config.test_fraction), stratify=frame.loc[trainval, "isFraud"], random_state=config.split_seed)
        parts = [train, val, test]
    for part in parts:
        if len(part) < 2 or frame.loc[part, "isFraud"].nunique() != 2:
            raise ValueError("Each split needs both classes. Increase sample size or change split fractions")
    assert not any(set(parts[i]) & set(parts[j]) for i in range(3) for j in range(i + 1, 3))
    ids = {key: frame.loc[idx, "TransactionID"].astype(int).tolist() for key, idx in zip(["train", "validation", "test"], parts)}
    split_hash = hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest()
    return parts, ids, split_hash

def summary(home, source):
    from ml.config import ExperimentConfig
    frame, meta = load_dataset(home, ExperimentConfig(dataset=source, max_rows=0))
    features = []
    for name in frame.columns:
        col = frame[name]
        categorical = name in KNOWN_CATEGORIES or not pd.api.types.is_numeric_dtype(col)
        info = {"name": name, "type": "categorical" if categorical else "numeric", "missing": int(col.isna().sum()), "unique": int(col.nunique())}
        if categorical:
            counts = col.fillna("missing").astype(str).value_counts().head(8)
            info["distribution"] = [{"name": str(k), "count": int(v)} for k, v in counts.items()]
        else:
            info["mean"] = None if col.isna().all() else float(col.mean())
            vals = col.dropna()
            counts, edges = np.histogram(vals, bins=10) if len(vals) else ([], [])
            info["distribution"] = [{"name": f"{edges[i]:.2g} to {edges[i+1]:.2g}", "count": int(v)} for i, v in enumerate(counts)]
            info["class_means"] = {str(int(k)): (None if pd.isna(v) else float(v)) for k, v in frame.groupby("isFraud")[name].mean().items()}
        features.append(info)
    return {**meta, "rows": len(frame), "columns": len(frame.columns), "fraud_count": int(frame.isFraud.sum()), "prevalence": float(frame.isFraud.mean()), "features": features}
