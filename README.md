# Synthetic OCT-A Blood Vessel Maps using Fundus Images and Generative Adversarial Networks

This repository provides an implementation for generating synthetic Optical Coherence Tomography Angiography (OCTA) images from retinal fundus images using a conditional GAN (Generative Adversarial Network) approach. The project leverages a pix2pix GAN architecture for translating input fundus images into synthetic OCTA images, designed to enhance data availability in medical imaging and research. This methodology aligns with the study conducted in *Scientific Reports* detailing OCTA synthesis through deep learning approaches: [Synthetic generation of OCT-A images using conditional GANs](https://www.nature.com/articles/s41598-023-42062-9).

Data can be accesed at (https://zenodo.org/records/6476639)

---

## Project Structure

### Key Components

1. **Data Preprocessing and Loading**
   - **`dataset_load_mod.py`** handles image loading, preprocessing, and patch extraction. The project splits images into smaller patches to facilitate training and final image reconstruction.
   
2. **Model Architecture**
   - **Generator** (`pix_generator.py`): This module contains a U-Net style encoder-decoder network with skip connections to retain spatial information.
   - **Discriminator** (`pix_discriminator.py`): Implements a PatchGAN classifier to discriminate between real and generated images at the patch level.

3. **Training Pipeline**
   - **Training Script** (`pix_pix_train.py`): Trains the pix2pix GAN on retinal fundus-to-OCTA translation. This script also includes data augmentation, logging, and checkpointing.

### Folder Structure

```
data/                      # Holds input images for training and testing
logs/                      # Stores training logs and TensorBoard summaries
training_checkpoints/      # Model checkpoints during training
pix_generator.py           # Generator network definition
pix_discriminator.py       # Discriminator network definition
pix_pix_train.py           # Main training script for pix2pix GAN
dataset_load_mod.py        # Data loading and patch processing
```

---

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy, Scikit-Image, Matplotlib, Pandas, SciPy


---

## Usage

### 1. Clone repository
```bash
git clone https://github.com/ivanco-uth/octa_synthesis.git
```

### 2. Download data from Zenodo into project folder

The project data can be accessed at: https://zenodo.org/records/6476639


### 3. Create & activate environment

```bash
conda create -n new_env python=3.7 -y
```

``` bash
conda activate new_env
```

### 4. Install requirements

```bash
pip install -r requirements.txt
```

### 5. Install conda libraries

```bash
conda install -c conda-forge cudatoolkit=11.0 cudnn=8.0.5
```

### 6. Train model

Run the training script as follows:

```bash
python pix_pix_train.py
```


### 7. Track metrics and visual outputs

Use Tensorboard to access metrics and outputs

```bash
tensorboard --logdir=logs/ --port='port_id'
```


---

## Evaluation and Results

During training, sample images and metrics are saved in the `logs/` directory. Visualization of training progression and evaluation metrics can be accessed via TensorBoard.

---

## Reference

Refer to the [study](https://www.nature.com/articles/s41598-023-42062-9) for a more detailed explanation of the model, methodology, and experimental setup, as well as to the `Synthetic_OCTA.pdf` document provided in this repository.
