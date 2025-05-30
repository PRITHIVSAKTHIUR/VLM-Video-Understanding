# **VLM-Video-Understanding**

> A minimalistic demo for image inference and video understanding using OpenCV, built on top of several popular open-source Vision-Language Models (VLMs). This repository provides Colab notebooks demonstrating how to apply these VLMs to video and image tasks using Python and Gradio.

## Overview

This project showcases lightweight inference pipelines for the following:
- Video frame extraction and preprocessing
- Image-level inference with VLMs
- Real-time or pre-recorded video understanding
- OCR-based text extraction from video frames

## Models Included

The repository supports a variety of open-source models and configurations, including:

- Aya-Vision-8B
- Florence-2-Base
- Gemma3-VL
- MiMo-VL-7B-RL
- MiMo-VL-7B-SFT
- Qwen2-VL
- Qwen2.5-VL
- Qwen-2VL-MessyOCR
- RolmOCR-Qwen2.5-VL
- olmOCR-Qwen2-VL
- typhoon-ocr-7b-Qwen2.5VL

Each model has a dedicated Colab notebook to help users understand how to use it with video inputs.

## Technologies Used

- **Python**
- **OpenCV** – for video and image processing
- **Gradio** – for interactive UI
- **Jupyter Notebooks** – for easy experimentation
- **Hugging Face Transformers** – for loading VLMs

## Folder Structure

```

├── Aya-Vision-8B/
├── Florence-2-Base/
├── Gemma3-VL/
├── MiMo-VL-7B-RL/
├── MiMo-VL-7B-SFT/
├── Qwen2-VL/
├── Qwen2.5-VL/
├── Qwen-2VL-MessyOCR/
├── RolmOCR-Qwen2.5-VL/
├── olmOCR-Qwen2-VL/
├── typhoon-ocr-7b-Qwen2.5VL/
├── LICENSE
└── README.md

````

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/PRITHIVSAKTHIUR/VLM-Video-Understanding.git
cd VLM-Video-Understanding
````

2. Open any of the Colab notebooks and follow the instructions to run image or video inference.

3. Optionally, install dependencies locally:

```bash
pip install opencv-python gradio transformers
```

## Hugging Face Dataset

The models and examples are supported by a dataset on Hugging Face:

[VLM-Video-Understanding](https://huggingface.co/datasets/prithivMLmods/VLM-Video-Understanding)

## License

This project is licensed under the Apache-2.0 License.
