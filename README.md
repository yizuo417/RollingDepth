# 🛹 RollingDepth: Video Depth without Video Models

[![Website](doc/badges/badge-website.svg)](https://rollingdepth.github.io)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face%20-Space-yellow)](https://huggingface.co/spaces/prs-eth/rollingdepth)
[![arXiv](https://img.shields.io/badge/arXiv-PDF-b31b1b)](http://arxiv.org/abs/2411.19189)
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-green)](https://huggingface.co/prs-eth/rollingdepth-v1-0)

This repository represents the official implementation of the paper titled "Video Depth without Video Models".

[Bingxin Ke](http://www.kebingxin.com/)<sup>1</sup>,
[Dominik Narnhofer](https://scholar.google.com/citations?user=tFx8AhkAAAAJ&hl=en)<sup>1</sup>,
[Shengyu Huang](https://shengyuh.github.io/)<sup>1</sup>,
[Lei Ke](https://www.kelei.site/)<sup>2</sup>,
[Torben Peters](https://scholar.google.com/citations?user=F2C3I9EAAAAJ&hl=de)<sup>1</sup>,
[Katerina Fragkiadaki](https://www.cs.cmu.edu/~katef/)<sup>2</sup>,
[Anton Obukhov](https://www.obukhov.ai/)<sup>1</sup>,
[Konrad Schindler](https://scholar.google.com/citations?user=FZuNgqIAAAAJ&hl=en)<sup>1</sup>


<sup>1</sup>ETH Zurich, 
<sup>2</sup>Carnegie Mellon University



## 📢 News
2024-12-02: Paper is on arXiv.<br>
2024-11-28: Inference code is released.<br>



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

# Install modified diffusers with cross-frame self-attention
bash script/install_diffusers_dev.sh 
```
We use [pyav](https://github.com/PyAV-Org/PyAV) for video I/O, which relies on [ffmpeg](https://www.ffmpeg.org/).


## 🏃 Test on your videos
All scripts are designed to run from the project root directory.

### 📷 Prepare input videos
1. Use sample videos:
    ```bash
    bash script/download_sample_data.sh
    ```

1. Or place your videos in a directory, for example, under `data/samples`.

### 🚀 Run with presets
```bash
python run_video.py \
    -i data/samples \
    -o output/samples_fast \
    -p fast \
    --save-npy true \
    --verbose
```
- `-p` or `--preset`: preset options
    - `fast` for **fast inference**, with dilations [1, 25] (flexible), fp16, without refinement, at max. resolution 768.
    - `fast1024` for **fast inference at resolution 1024**
    - `full` for **better details**, with dilations [1, 10, 25] (flexible), fp16, with 10 refinement steps, at max. resolution 1024.
    - `paper` for **reproducing paper numbers**, with (fixed) dilations [1, 10, 25], fp32, with 10 refinement steps, at max. resolution 768.
- `-i` or `--input-video`: path to input data, can be a single video file, a text file with video paths, or a directory of videos.
- `-o` or `--output-dir`: output directory.

Passing other arguments below may overwrite the preset settings:
- Coming soon
<!-- TODO: explain all arguments in detailed -->


## ⬇ Checkpoint cache
By default, the [checkpoint](https://huggingface.co/prs-eth/rollingdepth-v1-0) is stored in the Hugging Face cache. The HF_HOME environment variable defines its location and can be overridden, e.g.:

```
export HF_HOME=$(pwd)/cache
```

Alternatively, use the following script to download the checkpoint weights locally and specify checkpoint path by `-c checkpoint/rollingdepth-v1-0 `

```bash
bash script/download_weight.sh
```


## 🦿 Evaluation on test datasets
Coming soon


## 🎓 Citation
```bibtex
@misc{ke2024rollingdepth,
    title={Video Depth without Video Models}, 
    author={Bingxin Ke and Dominik Narnhofer and Shengyu Huang and Lei Ke and Torben Peters and Katerina Fragkiadaki and Anton Obukhov and Konrad Schindler},
    year={2024},
    eprint={2411.19189},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2411.19189}, 
}
```


## 🙏 Acknowledgments
We thank Yue Pan, Shuchang Liu, Nando Metzger, and Nikolai Kalischek for fruitful discussions. 
 
We are grateful to [redmond.ai](https://redmond.ai/) (robin@redmond.ai) for providing GPU resources.

## 🎫 License

This code of this work is licensed under the Apache License, Version 2.0 (as defined in the [LICENSE](LICENSE.txt)).

The model is licensed under RAIL++-M License (as defined in the [LICENSE-MODEL](LICENSE-MODEL.txt))

By downloading and using the code and model you agree to the terms in [LICENSE](LICENSE.txt) and [LICENSE-MODEL](LICENSE-MODEL.txt) respectively.
