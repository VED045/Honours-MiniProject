from typing import Literal
import hashlib
from pydantic import BaseModel, ConfigDict, Field, model_validator

class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    name: str = Field(default="ANN experiment", min_length=1, max_length=100)
    dataset: Literal["ieee-cis", "synthetic"] = "synthetic"
    expected_fingerprint: str | None = None
    run_role: Literal["primary", "exploratory", "demonstration"] = "exploratory"
    tree_iterations: int = Field(default=600, ge=10, le=3000)
    tree_depth: int = Field(default=7, ge=2, le=10)
    tree_learning_rate: float = Field(default=0.05, gt=0, le=0.5)
    tree_l2: float = Field(default=5, ge=0, le=100)
    tree_patience: int = Field(default=80, ge=10, le=300)
    tree_threads: int = Field(default=8, ge=1, le=20)
    tree_class_weighted: bool = False
    method: Literal["baseline", "smote", "cost_sensitive", "catboost"] = "baseline"
    hidden_layers: list[int] = Field(default=[64, 32], min_length=1, max_length=6)
    activation: Literal["relu", "gelu", "tanh", "elu"] = "relu"
    dropout: list[float] = Field(default=[0.2, 0.1])
    batch_norm: bool = False
    l1: float = Field(default=0, ge=0, le=1)
    l2: float = Field(default=0.0001, ge=0, le=1)
    optimizer: Literal["adam", "adamw", "sgd"] = "adam"
    learning_rate: float = Field(default=0.001, gt=0, le=1)
    batch_size: int = Field(default=256, ge=16, le=8192)
    epochs: int = Field(default=15, ge=1, le=500)
    patience: int = Field(default=5, ge=1, le=100)
    early_stopping: bool = True
    scheduler: Literal["none", "plateau"] = "plateau"
    gradient_clip: float = Field(default=5, ge=0, le=100)
    checkpoint_metric: Literal["average_precision", "roc_auc", "val_loss"] = "average_precision"
    loss: Literal["binary_cross_entropy"] = "binary_cross_entropy"
    output: Literal["sigmoid"] = "sigmoid"
    seed: int = Field(default=42, ge=0, le=2147483647)
    split_seed: int = Field(default=2026, ge=0, le=2147483647)
    split: Literal["temporal", "stratified"] = "temporal"
    validation_fraction: float = Field(default=0.15, gt=0, lt=0.4)
    test_fraction: float = Field(default=0.15, gt=0, lt=0.4)
    max_rows: int = Field(default=30000, ge=0)
    fraud_weight: float = Field(default=20, gt=0, le=1000)
    legitimate_weight: float = Field(default=1, gt=0, le=1000)
    smote_strategy: Literal["auto", "smote", "smotenc", "random"] = "auto"
    sampling_ratio: float = Field(default=0.3, gt=0, le=1)
    neighbors: int = Field(default=5, ge=1, le=50)
    threshold: float = Field(default=0.5, ge=0, le=1)
    optimize_threshold: bool = True
    selection_metric: Literal["f1", "balanced_accuracy", "recall", "precision"] = "f1"
    features: list[str] = Field(default_factory=list, max_length=500)
    exclude_features: list[str] = Field(default=["TransactionID", "TransactionDT"])
    categorical_features: list[str] = Field(default_factory=list)
    numerical_imputation: Literal["median", "mean", "constant"] = "median"
    categorical_imputation: Literal["constant", "most_frequent"] = "constant"
    encoding: Literal["onehot"] = "onehot"
    scaling: Literal["standard", "robust", "none"] = "standard"
    feature_selection: Literal["none", "variance"] = "none"
    bootstrap_samples: int = Field(default=50, ge=0, le=1000)
    max_memory_mb: int = Field(default=2048, ge=128, le=8192)

    @model_validator(mode="after")
    def validate_combinations(self):
        if any(n < 2 or n > 2048 for n in self.hidden_layers):
            raise ValueError("Each hidden layer must contain 2 to 2048 neurons")
        if len(self.dropout) != len(self.hidden_layers) or any(d < 0 or d >= 1 for d in self.dropout):
            raise ValueError("Provide one dropout in [0,1) per hidden layer")
        if self.validation_fraction + self.test_fraction >= 0.7:
            raise ValueError("At least 30% of observations must remain in training")
        if "isFraud" in self.features or "isFraud" in self.categorical_features:
            raise ValueError("Target cannot be a feature")
        if "TransactionID" in self.features:
            raise ValueError("TransactionID is a join key, not a predictor")
        if len(set(self.features)) != len(self.features):
            raise ValueError("Feature names must be unique")
        return self

    def config_hash(self):
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()
