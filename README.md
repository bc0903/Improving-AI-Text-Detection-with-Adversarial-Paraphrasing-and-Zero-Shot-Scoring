# RoBERTa-Based AI Text Detection with Zero-Shot and PPO Adversarial Training

This repository implements a hybrid AI-generated text detection framework combining:
- Supervised RoBERTa-based classification,
- Zero-shot perturbation stability scoring (DetectGPT-style),
- PPO-based adversarial paraphrasing for robustness testing.

## Structure

- `detect.ipynb`: **Main notebook** for running the full pipeline (training, evaluation, zero-shot scoring, adversarial paraphrasing).
- `ai_detector.py`: Supervised classification with RoBERTa (training, evaluation, prediction).
- `aigc_detect.py`: Script version of the pipeline for batch execution.
- `dataset_download.py`: Utilities to load and prepare datasets (e.g., HC3).
