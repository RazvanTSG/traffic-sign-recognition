# GTSRB Traffic Sign Recognition
**CNN Architecture Optimization & Interpretability (Grad-CAM)**

## Objective
The goal of this project is to design and optimize a Convolutional Neural Network (CNN) to classify 43 categories from the **German Traffic Sign Recognition Benchmark (GTSRB)**. The focus is on finding the optimal balance between **predictive accuracy** and **inference latency**.

## Engineering Methodology: Architecture Search
Instead of using a generic template, I conducted a systematic architecture search to identify the most efficient model for real-time deployment.



| Exp | Modification | Accuracy | Latency | Engineering Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Standard 2-Layer CNN | 96.6% | 1.0 ms | Fast, but lacks depth for complex features. |
| **2.2** | Added Conv (32 feat) | 98.5% | 1.0 ms | Improved feature extraction with zero overhead. |
| **2.5** | 64 filters + Pooling | 98.8% | 2.0 ms | High precision; acceptable latency trade-off. |
| **2.8** | Augmentation (Rot/Zoom) | 95.2% | 2.0 ms | **FAILED.** Orientation is a critical feature. |
| **2.9** | **Initial Kernel (5x5)** | **99.1%** | **2.0 ms** | **WINNER.** Larger initial receptive field. |

> **Critical Observation (Exp 2.8):** Implementing horizontal flips and heavy rotations significantly degraded performance. In traffic sign recognition, orientation is semantically meaningful (e.g., "Keep Left" vs. "Keep Right").

## Explainable AI (Grad-CAM Integration)
To ensure the model is not a "black box," I implemented a **Gradient-weighted Class Activation Mapping (Grad-CAM)** utility. This tool extracts feature maps from the final convolutional layer to visualize where the network focuses its attention during inference.



* **Logic:** Computes the average of the gradients of the predicted class with respect to the feature maps.
* **Result:** Validates that the model focuses on the central symbol rather than background noise or lighting artifacts.

## Repository Structure
* `traffic.py`: Training pipeline using the optimized (Exp 2.9) architecture.
* `gradcam.py`: Diagnostic tool for XAI (Explainable AI) visualization.
* `requirements.txt`: Environment dependencies.

## Setup & Usage

### 1. Dataset Setup
1. Download the GTSRB dataset from [Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).
2. Extract into a `data/` directory. Ensure the structure is `data/train/0...42/`.

### 2. Execution
```bash
# Train the model
python traffic.py data/ model.h5

# Visual Analysis (Grad-CAM)
python gradcam.py sample_sign.ppm model.h5
