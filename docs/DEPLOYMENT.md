# Deployment

## Current status
Local builds/tests are recorded in PROGRESS.md. No public frontend/backend deployment is claimed. The Vercel connector returned no team targets; the source repository is https://github.com/VED045/Honours-MiniProject; no Render account or persistent host was supplied. Credentials are never inferred or printed.

## Vercel frontend
Use a reviewed repository containing this project. In Vercel, select Root Directory = frontend, framework Next.js, Node 24, install npm ci, build npm run build. Set NEXT_PUBLIC_API_BASE_URL to the HTTPS backend origin in Preview and Production, and redeploy after changing it. This public variable contains only a URL, never an API key. frontend/vercel.json is a supported configuration format.

## Persistent Python host
Dockerfile and render.yaml prepare one paid persistent service for inference/artifact browsing. The API and optional training worker are separate processes on the same host via backend.supervisor, sharing the disk; Render disks cannot be shared across independent services. Default remote TRAINING_ENABLED=false avoids claiming that a small inference host can train the full dataset. Configure API_KEY privately (at least 24 characters), REQUIRE_API_KEY=true, and exact CORS_ORIGINS for approved frontend deployments. Supply the key in the dashboard's private connection field; it stays in memory, not NEXT_PUBLIC variables or exported files.

Before use, restore trusted artifacts/registry.sqlite3 plus completed run directories, data/processed aggregate dataset snapshot, and docs/references.json onto FRAUD_HOME's persistent disk. scripts/package_artifacts.py creates a consistent SQLite backup and an artifact archive without raw data or credentials. Extract to the persistent storage root. Missing artifacts return an explicit error. Empty registries remain functional; there are no invented model scores.

To enable cloud training, provision a sufficiently large persistent CPU host (at least comparable to the tested 16 GB machine for all raw columns), upload the licensed raw CSVs to data/raw/ieee-cis, verify them, and set TRAINING_ENABLED=true. The supervisor starts backend.worker as a persistent process. Training never runs inside Vercel frontend requests. Cloud interruption marks a stale running job failed after the heartbeat timeout; duplicate the configuration to retry.

For a VM, use separate API/worker processes under a service supervisor and one local durable disk; compose.yaml provides the local container arrangement. Independent scaled services require changing Store to a shared database and artifact operations to shared object storage. SQLite WAL is not a multi-host queue, and local disk assumptions must not be copied to ephemeral serverless filesystems.

## Smoke tests
Run python scripts/smoke_test.py --base-url https://your-backend --private-key-env API_KEY with the secret already configured in the environment. This checks health, models, configurations, aggregate summary, artifact/metrics consistency, and a real prediction with a registered model. It does not submit production training jobs. Confirm browser forms, progress and comparisons with the deployed API URL before assigning a production domain.

## Persistence, backups and operations
Back up metadata using SQLite's backup API plus immutable artifact directories. Keep raw Kaggle data private and comply with competition rules. Restrict artifacts to trusted operators because joblib deserialization is executable. Preserve fingerprints with each backup. Review queue failures, available disk/RAM and logs; do not collect secrets or raw feature payloads in logs.

## Troubleshooting
* CORS failure: exact protocol/host must be listed; localhost and 127.0.0.1 are separate origins.
* 401: enter the private backend key; do not publish it as a frontend env.
* Jobs stay queued: start worker and confirm the same FRAUD_HOME; API health does not claim worker readiness.
* 409 missing dataset/model: restore files/artifacts rather than generating mock responses.
* Out of memory: reduce rows/features or increase host RAM; guards cover dense matrices, not every raw parsing allocation.
* Lost results on redeploy: attach/restore a persistent disk at FRAUD_HOME.
* Notebook kernel missing: install/register the project kernel as described in notebooks/README.md.

Official verified documentation: https://vercel.com/docs/deployments and https://render.com/docs/disks (a paid disk attaches to one service), https://render.com/docs/docker and https://render.com/docs/background-workers.

## Deploy this GitHub repository, step by step

1. In Render, create a Blueprint from `VED045/Honours-MiniProject` using `render.yaml`. The paid persistent disk is required for saved models. Set a random private `API_KEY` (at least 24 characters), and initially set `CORS_ORIGINS` to the exact frontend origin when known. Keep `TRAINING_ENABLED=false` for an inference deployment.
2. On the trained local project run `python scripts/package_artifacts.py`. This creates `work/deployment/completed-artifacts.zip`, containing real completed models, probability arrays, the registry backup and aggregate summary; raw CSVs and credentials are excluded. Keep it private. Stop the Render API temporarily and upload this trusted archive to the host. Restore into its empty disk with `python scripts/restore_artifacts.py /path/to/completed-artifacts.zip --home /app/storage`. If the service already initialized an empty registry, restore into a fresh subdirectory (for example `/app/storage/research`) and set `FRAUD_HOME` to that directory before restarting; the script refuses to overwrite an existing registry.
3. In Vercel, import the same GitHub repository. Set Root Directory `frontend`, Framework `Next.js`, Node.js `24.x`, Install `npm ci`, Build `npm run build`. Add `NEXT_PUBLIC_API_BASE_URL=https://YOUR-RENDER-SERVICE.onrender.com`. Deploy.
4. Set Render `CORS_ORIGINS=https://YOUR-PROJECT.vercel.app` (comma-separated exact origins if needed) and restart the backend. Do not use `*` to bypass authentication/origin problems. Changing the frontend API origin requires a Vercel rebuild.
5. Open the deployed site, expand **Private backend connection**, and enter the backend key. Confirm the model registry, dataset scope, comparison curves and a saved-model prediction. Run `python scripts/smoke_test.py --base-url https://YOUR-RENDER-SERVICE.onrender.com --private-key-env API_KEY` with the key set privately in your shell.

Training from the hosted UI additionally requires the licensed raw dataset on the persistent host, sufficient RAM, and `TRAINING_ENABLED=true`. A frontend-only deployment can display the interface but cannot train or perform real inference until the backend and artifacts are connected.

GitHub Actions checks Python tests, TypeScript, ESLint, frontend tests, production build and the container build. Local Docker execution was unavailable on the build machine; CI provides that remaining environment check. No hosting subscription or public deployment was created automatically.
