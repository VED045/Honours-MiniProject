# Architecture and scientific protocol
Next.js browser → FastAPI typed request → SQLite WAL queue → separate PyTorch worker → immutable artifacts → inference/comparison API → dashboard, notebook and paper exporter.

HTTP never executes training. Worker claims are atomic; one job runs per worker. Heartbeats identify interrupted processes after ten minutes. Cancellation is checked every minibatch, while preprocessing/resampling finish their current stage first. UUID directories preserve prior results. Statuses are persisted and worker failures remain visible.

## Leakage controls
Training transaction/identity tables are left-joined one-to-one. Binary targets, unique nonmissing keys, identity membership and CRC integrity are verified. Unlabeled competition test data is never labeled evaluation data. Label-free exact covariate duplicate removal excludes TransactionID, TransactionDT and isFraud before partitioning; this does not guarantee entity isolation. Chronological boundaries keep equal relative times together. A deterministic limited sample is labeled development; model seeds do not change the shared split/sample seed.

All imputers/scalers/encoders fit training only. Known integer-coded categories are categorical. SMOTENC uses scaled numbers and ordinal codes, then one-hot encoding produces ANN inputs. Ordinary SMOTE is rejected when categories exist. Categorical-only auto chooses explicitly reported random oversampling. No holdout is resampled. Synthetic combinations can be implausible; distance geometry can weaken in high dimensions.

Baseline/SMOTENC BCE is unweighted. Cost-sensitive BCE multiplies fraud loss by w1 and legitimate loss by w0 without resampling. L1 is explicit; L2 configuration is optimizer weight decay (decoupled for AdamW). Weights are positive and are not probability cutoffs.

Checkpoint/early-stopping criteria and threshold search use validation only. Test labels do not tune. Interactive test-threshold inspection is post-hoc and warned about; validation is the UI default. AP uses sklearn average_precision_score, not trapezoidal integration. Curve compression affects display only. IID bootstrap intervals do not represent drift/dependence.

## Artifact v1
Each artifacts/UUID directory stores config.json, result.json, history.json, splits.json, predictions.npz, model.pt, preprocessor.joblib. result.json stores fingerprints/hashes, seeds, schema, encoded features, class counts, checkpoint, paths, metrics, curves, thresholds, timings and package versions. Predictions separate validation/test arrays. Dashboard, notebook, CSV and paper tables use the same format. A four-model runtime cache loads trusted artifacts only.

Integrated gradients uses 64 model-logit steps relative to zero encoded inputs; not causal, calibrated, or necessarily on-manifold. Inference rejects unknown/missing fields, reports unseen categories and cannot synthesize output when models are absent.

## Operational boundaries
Local API defaults to loopback. Remote deployment requires REQUIRE_API_KEY=true and a private API_KEY of at least 24 characters. CORS uses explicit origins. Requests and queue length are bounded. The frontend key remains in component memory and never appears in public envs/exports. This is a single-user prototype, not multi-tenant authorization. Dense-array guards do not include all raw CSV parsing memory.
