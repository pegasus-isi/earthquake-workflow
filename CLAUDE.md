# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a **Pegasus WMS workflow** for earthquake/seismic data analysis. It fetches data from the USGS Earthquake API and runs an 11-step parallel analysis pipeline via HTCondor. Scripts in `bin/` are standalone Python executables that can be run individually or orchestrated by Pegasus.

## Common Commands

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run individual analysis scripts directly
```bash
# Fetch data
./bin/fetch_earthquake_data.py --start-date 2024-01-01 --end-date 2024-01-31 --region california --min-magnitude 3.0 --output california_catalog.csv

# Run any analysis step (all follow --input / --output pattern)
./bin/analyze_seismic_patterns.py --input california_catalog.csv --output california_patterns.json
./bin/detect_seismic_anomalies.py --input california_catalog.csv --output california_anomalies.json
./bin/cluster_seismic_zones.py --input california_catalog.csv --output california_zones.json --method dbscan --eps 50 --min-samples 10
./bin/predict_aftershocks.py --input california_catalog.csv --output california_aftershock_predictions.json --mainshock-threshold 5.0
./bin/assess_seismic_hazard.py --input california_catalog.csv --output california_hazard.json --grid-resolution 0.5
./bin/analyze_seismic_gaps.py --input california_catalog.csv --output california_gaps.json --historical-years 20 --recent-years 5
```

### Generate and submit a Pegasus workflow
```bash
# Generate workflow DAG
./workflow_generator.py --regions california --start-date 2024-01-01 --end-date 2024-01-31 --min-magnitude 4.0 -o workflow.yml

# Submit to HTCondor via Pegasus
pegasus-plan --submit -s condorpool -o local workflow.yml

# Monitor
pegasus-status <submit_dir>
pegasus-analyzer <submit_dir>
```

### Build and push Docker container
```bash
cd Docker
docker build -f Earthquake_Dockerfile -t kthare10/earthquake-analysis:latest .
docker push kthare10/earthquake-analysis:latest
```

## Architecture

### Pipeline DAG (11 steps per region)

`fetch_earthquake_data` → parallel fan-out to:
- `analyze_seismic_patterns` (stats, b-value)
- `visualize_earthquakes` (maps/plots)
- `detect_seismic_anomalies` (swarms, rate changes)

Then from `analyze_seismic_patterns` + catalog:
- `cluster_seismic_zones` (DBSCAN/K-Means/Hierarchical)
- `predict_aftershocks` → `visualize_aftershock_predictions`
- `assess_seismic_hazard` → `visualize_seismic_hazard`
- `analyze_seismic_gaps` → `visualize_seismic_gaps`

Dependencies are inferred automatically by Pegasus via `infer_dependencies=True` based on file inputs/outputs.

### Key files

- **`workflow_generator.py`** — `EarthquakeWorkflow` class builds the Pegasus DAG (SiteCatalog, TransformationCatalog, ReplicaCatalog, Workflow). One set of 11 jobs is added per region.
- **`bin/`** — Standalone Python scripts; each is executable and takes `--input`/`--output` CLI args. Scripts produce CSV (raw data), JSON (analysis results), or PNG (visualizations).
- **`Docker/Earthquake_Dockerfile`** — Container used by Pegasus workers (Python 3.8 + pandas/numpy/matplotlib/scipy/scikit-learn).
- **`Access-Earthquake-workflow.ipynb`** — Jupyter notebook for running on ACCESS/FABRIC HPC resources.
- **`scratch/`** / **`output/`** — Pegasus working directories; outputs land in `output/` as `{region}_*.{csv,json,png}`.

### Execution environments

- **Local**: Run `bin/` scripts directly with Python
- **HTCondor/Pegasus**: Use `workflow_generator.py` to create the DAG, then `pegasus-plan --submit`
- **ACCESS / FABRIC**: Use the Jupyter notebook; requires Pegasus + HTCondor pre-configured on the cluster
- **Container**: Singularity pulls `docker://kthare10/earthquake-analysis:latest` on worker nodes

### Predefined regions

`pacific_ring`, `california`, `japan`, `indonesia`, `turkey`, `chile`, `worldwide`

### Data source

USGS FDSNWS Event API — no authentication required. Max 20,000 events per query; split date ranges for larger datasets.
