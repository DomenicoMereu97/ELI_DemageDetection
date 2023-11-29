## High-Energy Laser Damage Detection with Deep Learning
# Overview
High-energy laser facilities heavily rely on the integrity of optical components, and laser-induced damage resulting from thermodynamic effects during laser emission can lead to significant operational issues. The ability to detect changes in this damage is critical for understanding damage mechanisms, assessing damage resistance thresholds, and ensuring the safety and optimal performance of these facilities.

This repository presents a comprehensive study on damage detection in high-energy laser systems using three distinct deep learning approaches: ResNet, Siamese network, and Pseudo-Labeling. Additionally, we employ Grad-CAM for damage region identification. The choice of Grad-CAM is motivated by the absence of target labels denoting the specific locations of damage spots.

## Getting Started

1. **Clone the repository:**

    ```bash
    git clone https://github.com/DomenicoMereu97/ELI_DemageDetection.git
    ```

2. **Set up the environment:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Download the dataset and model checkpoint:**

   - Download the dataset and model checkpoint from [this Google Drive link](https://drive.google.com/drive/folders/1WkTzeLfyJlFnJbqbBc2MsUywNV080NV2?usp=sharing).
   - Place the dataset in the `data/` directory.
   - For the model checkpoint resnet.pth, create a `save/` directory and save the checkpoint file there.

4. **Explore the Jupyter notebooks:**

   - See Jupyter notebooks for model training, evaluation, and Grad-CAM visualization.

