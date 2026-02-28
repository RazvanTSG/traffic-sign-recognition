# GTSRB Traffic Sign Recognition & Architecture Optimization

**Objective:** Design, evaluate, and optimize a Convolutional Neural Network (CNN) for recognizing 43 categories of traffic signs from the German Traffic Sign Recognition Benchmark (GTSRB).

## Engineering Methodology (Architecture Search)
Instead of relying on a standard template, I conducted a systematic architecture search focusing on the trade-off between **Accuracy** and **Inference Latency (ms/step)**.

| Exp | Modification | Accuracy | Inference Latency | Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| Baseline | Basic CS50 Architecture | 96.6% | 1.0 ms | Fast, but struggles with nuanced signs. |
| 2.2 | Added Conv Layer (32 filters, no pool) | 98.5% | 1.0 ms | Better feature extraction without speed penalty. |
| 2.5 | Increased to 64 filters + pooling | 98.8% | 2.0 ms | Excellent accuracy, acceptable latency hit. |
| 2.8 | Added Rotation/Zoom Augmentation | 95.2% | 2.0 ms | Performance dropped. *Observation: Traffic signs are orientation-dependent (e.g., Left vs Right turn).* |
| **2.9** | **Changed initial Kernel to (5,5)** | **99.1%** | **2.0 ms** | **Optimal Setup.** Wider initial receptive field captures sign shapes better. Loss: 0.0414. |

## Explainable AI (Grad-CAM Integration)
To verify that the CNN is learning relevant features (e.g., the symbol on the sign) and not just background noise, I implemented a **Gradient-weighted Class Activation Mapping (Grad-CAM)** script.

The `gradcam.py` tool extracts the feature maps from the final convolutional layer and overlays an attention heatmap onto the original image.

## Repository Structure
* `traffic.py`: The training pipeline with the optimized (Exp 2.9) CNN architecture.
* `gradcam.py`: Diagnostic tool for visualizing the network's attention regions.
* `requirements.txt`: Environment dependencies.

## Dataset Setup
The GTSRB dataset is not included in this repository due to size constraints. 
To run this project:
1. Download the dataset from [Kaggle - GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).
2. Extract the contents into a folder named `data/` in the root directory.
3. Ensure the structure is `data/train/0...42/`.

## Usage
**1. Train the model:**
```bash
python traffic.py data_directory model.h5
