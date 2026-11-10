# NAVSIM

## Environment

Use Linux, Python 3.10 and a CUDA-capable PyTorch installation for Epona inference.

```bash
conda create -n foresight python=3.10 -y
conda activate foresight
bash scripts/setup_env.sh
bash scripts/test.sh
```

The setup script installs requirements, [nuplan-devkit](https://github.com/motional/nuplan-devkit)
(v1.2), and this project into the active environment. It needs internet access and Git.
The pinned PyTorch 2.1 / torchvision 0.16 pair requires a compatible NVIDIA driver
for GPU inference. Prepare the [NAVSIM](https://github.com/autonomousvision/navsim)
OpenScene logs, sensor blobs and maps through their official distribution channels.
The expected data layout is `navsim_logs/{trainval,test}` and
`sensor_blobs/{trainval,test}` under `OPENSCENE_DATA_ROOT`.

Copy `.env.example` to `.env`, set the asset paths, then run `source .env`.
Provide the native-rate Epona NuPlan checkpoint, temporal DCAE checkpoint,
ResNet-34 backbone weights and trajectory anchors. For training, an optional
planner initialization checkpoint can be supplied with `FORESIGHT_PLANNER_PRETRAIN`.

```bash
cp .env.example .env  # First setup only; edit all /path/to values before sourcing.
source .env
```

Use the NAVSIM v1 data/splits matching this repository. Obtain OpenScene data and
nuPlan maps from their official distribution channels; the retained `download/`
scripts are optional upstream download helpers. A typical directory layout is:

```text
$OPENSCENE_DATA_ROOT/
  navsim_logs/{trainval,test}/
  sensor_blobs/{trainval,test}/
  maps/nuplan-maps-v1.0/
```

Set the four asset paths `FORESIGHT_EPONA_CKPT`, `FORESIGHT_VAE_CKPT`,
`FORESIGHT_BACKBONE_CKPT` and `FORESIGHT_PLAN_ANCHOR` in `.env`.
`FORESIGHT_PLANNER_PRETRAIN` is optional. Set `FORESIGHT_EVAL_CKPT` to the ForeSight checkpoint to evaluate.

## Commands

```bash
bash scripts/check_data.sh train
bash scripts/cache_dataset.sh
BATCH_SIZE=8 NPROC_PER_NODE=1 TRAIN_EPOCHS=100 bash scripts/train.sh
# After training: set FORESIGHT_EVAL_CKPT in .env and source .env again.
bash scripts/check_data.sh evaluate
bash scripts/cache_metrics.sh
bash scripts/evaluate.sh
```

Run commands from the repository root. Dataset caching prepares training features
without loading world-model weights; metric caching prepares navtest PDM inputs.
`check_data.sh` checks nonempty directories and asset-file existence, not dataset
completeness or checkpoint compatibility. `BATCH_SIZE` is per training process.
Training runs the full future-conditioned model; this entry point does not
implement the paper's separate planner-only pretraining stage.

Use `NNODES`, `NPROC_PER_NODE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`,
`BATCH_SIZE` and `TRAIN_EPOCHS` to configure the launcher. Hydra overrides may
be appended to each command. `FORESIGHT_DRY_RUN=1` prints the command without
loading models or starting a job.

## Temporal configuration

`agent.config.video_fps` defaults to 5 Hz. The trajectory interval defaults
to 0.5 seconds and its horizon to 4 seconds. The native-rate world model therefore
generates 20 future frames while the planner predicts 8 trajectory states.
Set `video_fps` to the actual sampling rate of the Epona checkpoint; changing
this setting does not retime the pretrained world model.
Timestamps are measured in seconds from the current observation. Each action
state can attend to all video tokens within the horizon.

The number of video queries per frame is controlled by
`agent.config.video_queries_per_frame`. The resampler supports variable frame
counts and uses shared frame queries with continuous sinusoidal time encoding.

Model inference requires current sensor features only. Future trajectories are
used as training targets and are not inputs to world-model generation or evaluation.

Evaluation loads the complete checkpoint strictly. Planner initialization loads
only compatible tensors and logs their count. A checkpoint must match the model
architecture used for evaluation.

## Tests

```bash
bash scripts/test.sh
```

The tests run on CPU without downloading model weights or data, and do not need
a configured `.env`. Missing optional runtime dependencies are reported as skipped
integration tests. This is a structural check; it does not measure NAVSIM PDMS
or full-checkpoint GPU performance.

Upstream license texts are retained in [LICENSES](../LICENSES/README.md).
