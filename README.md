# EDTH Hackathon London 2025

Description of the hackathon project for the European Defence Tech Hackathon in London 2025.

See [this information spreadsheet](https://docs.google.com/spreadsheets/d/1AT8ndsEe9hgljTUFgmpAJ_pYE9kLxrVdJI-xuT68uto/edit?referrer=luma&gid=2054763765#gid=2054763765).
Data is available through [this sharepoint](https://quantumdrones-my.sharepoint.com/personal/pzimmermann_quantum-systems_com/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fpzimmermann%5Fquantum%2Dsystems%5Fcom%2FDocuments%2F2025%5FLondon%5FHackathon&ga=1).

## Project Focus

Tackling Quantum-6 (and Stark-1 to a degree). We aim to:

1. Identify objects in video in real time.
2. Estimate velocity, acceleration, and class for those objects.
3. Maintain a live map that predicts where previously identified objects are likely to be.
4. Recognise when an object that has been seen before re-enters the scene.

## Setup and Running the Project

This project uses Poetry for dependency management and has a critical version requirement for one of its packages.

### 1. Environment Setup

It is highly recommended to use a Python virtual environment.

```bash
# Create and activate a virtual environment (example using venv)
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

Install the dependencies listed in `pyproject.toml` using Poetry.

```bash
poetry install
```

### 3. Resolve Dependency Conflict (Crucial Step)

There is a known incompatibility between the `trackers` and `supervision` packages. To fix this, you must install `trackers` and downgrade `supervision` to a specific version. **The scripts will not run without this step.**

```bash
pip install trackers
pip install supervision==0.21.0
```

### 4. Place Video Files

Place the video files you want to process inside the `data/` directory.

### 5. Run the Tracking Script

To process a video file from the command line, use the manual runner in the backend package:

```bash
python backend/tracking/trackers_in_video.py data/your_video_name.mp4
```

The processed video will be saved alongside the input with `_output.mp4` appended to its filename. Existing outputs are rotated to `*_output_old.mp4` automatically so you never lose the previous run.
