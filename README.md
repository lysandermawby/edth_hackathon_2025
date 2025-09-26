# EDTH Hackathon London 2025

Description of the hackathon project for the European Defence Tech Hackathon in London 2025.

See [this information spreadsheet](https://docs.google.com/spreadsheets/d/1AT8ndsEe9hgljTUFgmpAJ_pYE9kLxrVdJI-xuT68uto/edit?referrer=luma&gid=2054763765#gid=2054763765).

Still very much finding projects of interest. Will move cleaned notes from workshops into this.

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

First, install the dependencies listed in `pyproject.toml` using Poetry.

```bash
# Install base dependencies
poetry install
```

### 3. Resolve Dependency Conflict (Crucial Step)

There is a known incompatibility between the `trackers` and `supervision` packages. To fix this, you must install `trackers` and downgrade `supervision` to a specific version. **The scripts will not run without this step.**

```bash
# Install the required 'trackers' package
pip install trackers

# Downgrade 'supervision' to the compatible version
pip install supervision==0.21.0
```

### 4. Place Video Files

Place the video files you want to process inside the `/data` directory.

### 5. Run the Tracking Script

To process a video file, run the `trackers_test.py` script from the project's root directory. You must provide the path to the video file as an argument.

```bash
# Example of running the script
python src/tracking/trackers_test.py data/your_video_name.mp4
```

The processed video will be saved in the same directory as the input video with `_output.mp4` appended to its name.

