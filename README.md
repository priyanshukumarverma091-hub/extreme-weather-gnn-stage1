# Extreme Weather GNN (Stage 1)

Graph neural network that **detects extreme-weather events** on a ~12 km NWP grid and
**tracks them through time**. This is Stage 1 of a two-stage research pipeline; Stage 2
(diffusion-based downscaling, 12 km → 5 km) is planned and not part of this repository yet.

## What it does

1. Turns the lat/lon grid into a graph (nodes = grid cells, edges = spatial neighbours with
   great-circle / geometric edge features).
2. Runs an edge-conditioned graph-attention network (`ExtremeWeatherGNN`, built from
   `GeodesicAttentionLayer` blocks) on 16 input features per node.
3. Trains with a multi-task loss: detection, localisation, extent and confidence.
4. Selects the decision threshold on the **validation** set only, then evaluates once on test.
5. Extracts events (connected components above the threshold, minimum size filter) and tracks
   them across timesteps with Hungarian matching and a maximum-distance limit.
6. Writes probability maps, binary masks, event metadata, trajectories and a NetCDF file.

Pure PyTorch implementation (no DGL / PyTorch Geometric dependency).

## Repository layout

```
.
├── stage1_gnn.py        # full pipeline: train -> evaluate -> infer -> track
├── inspect_outputs.py   # prints a summary of the saved outputs
├── requirements.txt
├── data/
│   └── README.md        # expected NetCDF format (data files are not in the repo)
└── .gitignore
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA GPU is strongly recommended for training; the script falls back to CPU otherwise.

## Data

Put `train.nc`, `val.nc` and `test.nc` in `data/`. Variable names, dimensions and the 16-feature
input layout are described in [`data/README.md`](data/README.md).

## Run

```bash
python stage1_gnn.py --data-dir data --output-dir outputs
python inspect_outputs.py --output-dir outputs/stage1_outputs
```

Options:

| Flag | Default | Meaning |
|---|---|---|
| `--data-dir` | `data` | Folder with `train.nc`, `val.nc`, `test.nc` |
| `--output-dir` | `outputs` | Checkpoints, results JSON, event outputs |
| `--prob-threshold` | `0.88` | Threshold for inference-time event extraction (use the validation-selected value from your training run) |

Hyper-parameters (hidden size, layers, heads, learning rate, epochs, loss weights) are constants
in the *Configuration* section near the top of `stage1_gnn.py`. Seed is fixed to 42.

## Outputs

```
outputs/
├── extreme_weather_stage1_best.pt      # best checkpoint (by validation F1)
├── extreme_weather_stage1_final.pt
├── stage1_results.json                 # metrics
└── stage1_outputs/
    ├── event_probability_maps.npy
    ├── event_masks.npy
    ├── stage1_events.json              # per-timestep events (centroid, bbox, probabilities)
    ├── event_trajectory.json           # tracked centroid path with displacement
    └── stage1_event_maps.nc            # probability + mask as NetCDF
```

Large files (`.nc`, `.pt`, `.npy`) are git-ignored. Share them via GitHub Releases, Git LFS,
Zenodo or a cloud-storage link.

## Results

_Add your validation/test metrics here (precision, recall, F1, ROC-AUC, PR-AUC)._

## Status / known limitations

- The script was exported from a notebook: it runs the whole pipeline in one process and is not
  yet split into importable modules.
- Inference reuses the model and graph built earlier in the same run.
- Diffusion-based Stage 2 is not implemented here.

## License

_Add a license (e.g. MIT) via GitHub: Add file → Create new file → `LICENSE`._
