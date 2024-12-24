

## 🛠️ Setup
The inference code was tested on: Debian 12, Python 3.12.7 (venv), CUDA 12.4, GeForce RTX 3090

### 📦 Repository
```bash
git clone https://github.com/prs-eth/RollingDepth.git
cd RollingDepth
```

### 🐍 Python environment
Create python environment:
```bash
# with venv
python -m venv venv/rollingdepth
source venv/rollingdepth/bin/activate

# or with conda
conda create --name rollingdepth python=3.12
conda activate rollingdepth
```

### 💻 Dependencies
Install dependicies: 
```bash
pip install -r requirements.txt
bash script/install_diffusers_dev.sh  # Install modified diffusers with cross-frame self-attention
```
We use [pyav](https://github.com/PyAV-Org/PyAV) for video I/O, which relies on [ffmpeg](https://www.ffmpeg.org/) (tested with version 5.1.6-0+deb12u1).

To see the modification in diffusers, search for comments "Modified in RollingDepth".

## 🏃 Test on your videos
All scripts are designed to run from the project root directory.

### 📷 Prepare input videos


1.  place your videos in a directory, for example, under `data/fg` and `data/fg`.

### 🚀 Run with presets
```bash
python run_video.py \
    -fi data/fg \
    -bi data/fg \
    -o output/ iclight \
    -p fast1024 \
    --save-npy False \
    --save-sbs False \
    --verbose
```
- `-p` or `--preset`: preset options
    - `fast` for **fast inference**, with dilations [1, 25] (flexible), fp16, without refinement, at max. resolution 768.
    - `fast1024` for **fast inference at resolution 1024**
    - `full` for **better details**, with dilations [1, 10, 25] (flexible), fp16, with 10 refinement steps, at max. resolution 1024.
    - `paper` for **reproducing paper numbers**, with (fixed) dilations [1, 10, 25], fp32, with 10 refinement steps, at max. resolution 768.
- `-fi` or `--input-fg-video`: path to input data, can be a single video file, a text file with video paths, or a directory of videos.
- `-bi` or `--input-bg-video`: path to input data, can be a single video file, a text file with video paths, or a directory of videos.
- `-o` or `--output-dir`: output directory.

#### Passing these inference arguments will overwrite the preset settings:
- `--res` or `--processing-resolution`: the maximum resolution (in pixels) at which image processing will be performed. If set to 0, processes at the original input image resolution.
- `--refine-step`: number of refinement iterations to improve accuracy and details. Set to 0 to disable refinement.
- `--snip-len` or `--snippet-lengths`: number of frames to analyze in each snippet.
- `-d` or `--dilations`: spacing between frames for temporal analysis, could have multiple values e.g. `-d 1 10 25`.

#### Clip sub-sequence to be processed:
- `--from` or `--start-frame`: the starting frame index for processing, default to 0.
- `--frames` or `--frame-count`: number of frames to process after the starting frame. Set to 0 (default) to process until the end of the video.

#### Output settings
- `--fps` or `--output-fps`: frame rate (FPS) for the output video. Set to 0 (default) to match the input video's frame rate.
- `--restore-res` or `--restore-resolution`: whether to restore the output to the original input resolution after processing, Default: False.
- `--save-sbs` or `--save-side-by-side`: whether to save side-by-side videos of RGB and colored depth. Default: True.
- `--save-npy`: whether to save depth maps as .npy files. Default: True.
- `--save-snippets`: whether to save initial snippets. Default: False

#### Other argumenets
- Please run `python run_video.py --help` to get details for other arguments.
- For low GPU memory footage:  pass ` --max-vae-bs 1 --unload-snippet true` and use a smaller resolution, e.g. `--res 512`







## 🎫 License

This code of this work is licensed under the Apache License, Version 2.0 (as defined in the [LICENSE](LICENSE.txt)).

The model is licensed under RAIL++-M License (as defined in the [LICENSE-MODEL](LICENSE-MODEL.txt))

By downloading and using the code and model you agree to the terms in [LICENSE](LICENSE.txt) and [LICENSE-MODEL](LICENSE-MODEL.txt) respectively.
