import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE, SMOTENC, RandomOverSampler
from ml.data import KNOWN_CATEGORIES

class Preprocessor:
    def __init__(self, config):
        self.config = config

    def fit(self, frame):
        excluded = set(self.config.exclude_features) | {"isFraud", "TransactionID"}
        self.features = self.config.features or [c for c in frame if c not in excluded]
        unknown = set(self.features + self.config.categorical_features) - set(frame)
        if unknown:
            raise ValueError(f"Unknown features: {sorted(unknown)}")
        if not self.features or set(self.features) & {"isFraud", "TransactionID"}:
            raise ValueError("Select predictors excluding target and identifier")
        self.cat = [c for c in self.features if c in KNOWN_CATEGORIES or c in self.config.categorical_features or not pd.api.types.is_numeric_dtype(frame[c])]
        self.num = [c for c in self.features if c not in self.cat]
        self.num_imputer = SimpleImputer(strategy=self.config.numerical_imputation, fill_value=0, keep_empty_features=True)
        self.cat_imputer = SimpleImputer(strategy=self.config.categorical_imputation, fill_value="missing", keep_empty_features=True)
        self.scaler = {"standard": StandardScaler, "robust": RobustScaler}.get(self.config.scaling, lambda: None)()
        if self.num:
            numeric = self.num_imputer.fit_transform(frame[self.num])
            if self.scaler:
                self.scaler.fit(numeric)
        if self.cat:
            categorical = self.cat_imputer.fit_transform(self._categorical(frame))
            self.ordinal = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            self.ordinal.fit(categorical)
            codes = self.ordinal.transform(categorical)
            self.onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
            self.onehot.fit(codes)
        self.selector = None
        return self

    def _categorical(self, frame):
        return frame[self.cat].astype(object).where(frame[self.cat].notna(), np.nan).map(lambda v: str(v) if pd.notna(v) else np.nan)

    def base(self, frame):
        if set(self.features) - set(frame):
            raise ValueError("Prediction schema is missing required features")
        chunks = []
        if self.num:
            numeric = self.num_imputer.transform(frame[self.num])
            chunks.append(self.scaler.transform(numeric) if self.scaler else numeric)
        if self.cat:
            chunks.append(self.ordinal.transform(self.cat_imputer.transform(self._categorical(frame))))
        return np.concatenate(chunks, axis=1).astype(np.float32)

    def encode(self, base):
        width = len(self.num) + (sum(len(c) for c in self.onehot.categories_) if self.cat else 0)
        if len(base) * width * 4 > self.config.max_memory_mb * 1024**2:
            raise MemoryError("Dense representation exceeds configured memory budget; reduce rows/features")
        chunks = [base[:, :len(self.num)]] if self.num else []
        if self.cat:
            chunks.append(self.onehot.transform(base[:, len(self.num):]))
        return np.concatenate(chunks, axis=1).astype(np.float32)

    def transform(self, frame):
        values = self.encode(self.base(frame))
        return self.selector.transform(values) if self.selector else values

    def training(self, frame, y):
        base = self.base(frame)
        before = np.bincount(y.astype(int), minlength=2).tolist()
        strategy = "none"
        if self.config.method == "smote":
            strategy = self.config.smote_strategy
            if strategy == "auto":
                strategy = "smotenc" if self.cat and self.num else ("random" if self.cat else "smote")
            if strategy == "smote" and self.cat:
                raise ValueError("SMOTE requires numerical features; use SMOTENC or random oversampling for categories")
            if strategy == "smotenc" and (not self.cat or not self.num):
                raise ValueError("SMOTENC requires both numerical and categorical predictors")
            if before[1] >= before[0] or self.config.sampling_ratio <= before[1] / before[0]:
                raise ValueError("Oversampling ratio must exceed original fraud/legitimate ratio; fraud must be minority")
            if strategy != "random" and before[1] <= self.config.neighbors:
                raise ValueError("Fraud training count must exceed neighbor count")
            projected = int(before[0] * (1 + self.config.sampling_ratio))
            # Check encoded allocation before oversampler allocates synthetic arrays.
            width = len(self.num) + (sum(len(c) for c in self.onehot.categories_) if self.cat else 0)
            if projected * width * 4 * 3 > self.config.max_memory_mb * 1024**2:
                raise MemoryError("Resampling would exceed configured memory budget")
            kw = dict(sampling_strategy=self.config.sampling_ratio, random_state=self.config.seed)
            if strategy == "random":
                sampler = RandomOverSampler(**kw)
            elif strategy == "smotenc":
                sampler = SMOTENC(categorical_features=list(range(len(self.num), base.shape[1])), k_neighbors=self.config.neighbors, **kw)
            else:
                sampler = SMOTE(k_neighbors=self.config.neighbors, **kw)
            base, y = sampler.fit_resample(base, y)
        encoded = self.encode(base)
        if self.config.feature_selection == "variance":
            self.selector = VarianceThreshold()
            encoded = self.selector.fit_transform(encoded)
        self.feature_names = self.num + (self.onehot.get_feature_names_out(self.cat).tolist() if self.cat else [])
        if self.selector:
            self.feature_names = np.array(self.feature_names)[self.selector.get_support()].tolist()
        return encoded, np.asarray(y), {"original": before, "resampled": np.bincount(y.astype(int), minlength=2).tolist(), "strategy": strategy}
