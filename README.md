# Extreme Weather GNN (Stage 1)

Graph attention network that detects extreme weather events on a ~12 km
NWP grid and tracks them across timesteps. Stage 1 of a two-stage pipeline
(Stage 2: diffusion-based downscaling, 12 km → 5 km, in progress).

## Features
- Grid → graph with geodesic (spherical) attention layers
- Multi-task loss: detection, localization, extent, confidence
- Validation-only threshold selection
- Connected-component event extraction + Hungarian-matching tracking
- Outputs: probability maps, event masks, trajectories (JSON), NetCDF

## Input data (train.nc / val.nc / test.nc)
Dims: time, latitude, longitude
Variables: u10_norm, v10_norm, t2m_norm, msl_norm, tp_norm, cape_norm,
           orography, land_sea_mask, efi, extreme_mask

## Setup
pip install -r requirements.txt

## Usage
python stage1_gnn.py --train ... --val ... --test ...

## Outputs
stage1_event_maps.nc, stage1_events.json, event_trajectory.json, ...

## Results
(apne actual validation/test numbers yahan daalna)

## License
