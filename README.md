# Multi-GPU Raster Inference Demo

This repository contains a fairly simple proof of concept for multi-GPU inference over a PyTorch dataloader. Most of the complexity is in ensuring that your environment is set up correctly, which this codebase can be used to check. It can be modified for use with different dataloaders and/or models.

A basic dataloader is provided to load tiles from a given raster (typically an orthomosaic'd GeoTiff), or you can generate dummy data for testing with a simulated delay. This enables you to easily see speedups from using multi-GPU machines, which should typically provide a close-to linear speedup in throughput.

All distributed logic is handled by PyTorch Lightning and CLI configuration is via Hydra. So, while training is currently out of scope, there's no reason why you couldn't add a training (`fit`) step.

## Setup

Dependencies are specified in `pyproject.toml` and we recommend using `uv` for installation. Rasterio depends on `gdal` which is a pain to install on most systems, however you can see how far you by running `uv sync`. This should install two scripts in your path: `mgpr` (Multi-GPu-Raster) and `mgpr-test-image`.

## Usage

One of the goals of this repo was to troubleshoot dataloading and inference on SLURM clusters, so we provide a simple job that can be configured to your target system. You might find that just running the main script "just works" on a local, multi-GPU machine.

### Locally

You can run the code as `mgpr`:

```
uv run mgpr gpus=2 batch_size=16 input_path=test_image.tiff# etc
```

Lightning uses the `devices` flag (set here to `gpus`) to determine how many [devices to use](https://lightning.ai/docs/pytorch/stable/common/trainer.html#devices); it's not the individual device IDs, so 2 == 2 GPUs. Depending on your system you can play with `batch_size`.

The script defaults to using a dummy dataloader that returns a fixed number of tiles of a given size. You can control this with:

```
use_fake_data: true
dummy_dataset_delay: 0.01
dummy_dataset_count: 10000
tile_size: 1024
```

which, for this example, will return a dataset of length 10000, yielding 1024x1024x3 px tiles (torch tensors). This is useful for simple environment testing. Alternatively, the script can use an input image provided in the config (`image_path`). We provide a utility called `mgpr-test-image ` that uses the `image_size` setting to create a randomly filled GeoTIFF on disk that can be used to simulate actual dataloading.

### SLURM

On SLURM the main difference we need to carefully tell the manager what to provision and how many processes etc we request. The key parameters are:

```
gpus: N
ntasks-per-node: N
```

The number of tasks is critical, because DDP spawns an additional process on the system. If you don't do this, processing will take up to N times as long. If execution continues correctly, you'll see something like this in your `stderr`:

```
Using 16bit Automatic Mixed Precision (AMP)
Using default `ModelCheckpoint`. Consider installing `litmodels` package to enable `LitModelCheckpoint` for automatic upload to the Lightning model registry.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/2
Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/2
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All distributed processes registered. Starting with 2 processes
----------------------------------------------------------------------------------------------------

LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1]
LOCAL_RANK: 1 - CUDA_VISIBLE_DEVICES: [0,1]
SLURM auto-requeueing enabled. Setting signal handlers.
SLURM auto-requeueing enabled. Setting signal handlers.
```

otherwise if you only see a single rank reporting, this suggests that you're deadlocked somewhere as the system is waiting for all ranks to be created. It's also important that the job uses `srun` which handles parallel task generation. If you run the script directly, it won't work.

To launch a job on your own cluster, run:

```
export NUM_GPUS=2; sbatch --ntasks-per-node=$NUM_GPUS --gpus=$NUM_GPUS job.slurm
```

If your cluster supports it, try to constrain which GPU will run e.g. `--gpus=a100:$NUM_GPUS` to give more consistent results. For example, on some clusters (including HiPerGator) you will get different GPUs for single vs multi machines.

The script will log to a `slurm_logs` folder, by job ID. To check while running:

```
tail -f slurm_logs/<job id>.out
```

You should find that on a good GPU, runtime will be fast - under 5 minutes. If you find the process is timing out, check that there hasn't been an error somewhere.

Modify Hydra parameters in the job file!

## Models

Two models are included - a ResNet50 and FasterRCNN. They are configured with default pretrained weights, but are purely for testing purposes. You can have a look at the `models.py` file to add your own, or to load in your own weights if you need better timing estimates. The only thing that's non-standard is a subclass of LightningModule that will periodically profile GPU memory usage during prediction.

## Verification

With everything set up, note here we're requesting fixed GPU types for comparison:

```
export NUM_GPUS=1; sbatch --ntasks-per-node=$NUM_GPUS --gpus=a100:$NUM_GPUS job.slurm
export NUM_GPUS=2; sbatch --ntasks-per-node=$NUM_GPUS --gpus=a100:$NUM_GPUS job.slurm
```

And the results should look something like (note the very small utilisation here as the batch size is tiny)

```
# Single A100
Predicting DataLoader 0: 100%|██████████| 417/417 [01:38<00:00,  4.23it/s]
GPU Usage Summary:
GPU 0:
  Average Memory Usage per Batch: 0.53 GB (0.67%)
  Peak Memory Usage: 6.66 GB (8.42%)
  Average Utilization per Batch: 0.67%
[2025-05-09 18:50:53,428][root][INFO] - Prediction completed in 101.64 seconds
```

```
# Dual A100
Predicting DataLoader 0: 100%|██████████| 209/209 [00:50<00:00,  4.16it/s]
GPU Usage Summary:
GPU 0:
  Average Memory Usage per Batch: 0.53 GB (0.67%)
  Peak Memory Usage: 6.66 GB (8.42%)
  Average Utilization per Batch: 0.67%
GPU 1:
  Average Memory Usage per Batch: 0.53 GB (0.67%)
  Peak Memory Usage: 6.66 GB (8.42%)
  Average Utilization per Batch: 0.67%
[2025-05-09 18:50:11,379][root][INFO] - Prediction completed in 54.90 seconds
[2025-05-09 18:50:11,413][root][INFO] - Prediction completed in 55.10 seconds
```

 Logging here is printing for all ranks (i.e. 2 GPUs). You can then experiment with numbers of CPUs, workers etc. You will find that the more powerful a system you specify, the longer it will take for the job to be queued. Also be careful to allow 2xCPUs for a given number of workers as these will be spawned per rank.
