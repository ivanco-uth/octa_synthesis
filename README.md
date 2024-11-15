# Synthetic OCT-A blood vessel maps using fundus images and generative adversarial networks

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
data_slice.py              # Utility for patch operations and image reconstruction
```

---

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy, Scikit-Image, Matplotlib, Pandas, SciPy

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Prepare the Dataset

Organize your data folder with fundus and OCTA images in a structure compatible with `dataset_load_mod.py`.

### 2. Train the Model

Run the training script as follows:

```bash
python pix_pix_train.py --case_range <range> --gpu_id <gpu_id>
```

Arguments:

- `--case_range`: Specifies the range of test cases.
- `--gpu_id`: Specifies the GPU for running the script.

### 3. Generate Synthetic OCTA Images

After training, modify `pix_pix_train.py` to use the `test_set` and execute:

```bash
python pix_pix_train.py
```

### Model Details

#### Generator

The generator uses a U-Net architecture, comprising encoding and decoding layers with skip connections for spatial feature retention. It outputs the translated OCTA image.

```python
def Generator(in_channels, patch_dim, out_channels):
    # Define model layers and forward pass
```

#### Discriminator

The PatchGAN discriminator evaluates the quality of generated images by examining patches, ensuring local-level consistency.

```python
def Discriminator(in_channels, out_channels, patch_dim):
    # Define model layers and forward pass
```

---

## Evaluation and Results

After training, sample results and metrics are saved in the `logs/` directory. Visualization of training progression and evaluation metrics (such as MSE and SSIM) can be accessed via TensorBoard.

---

## Reference

Refer to the [study](https://www.nature.com/articles/s41598-023-42062-9) for a more detailed explanation of the model, methodology, and experimental setup, as well as to the `Synthetic_OCTA.pdf` document provided in this repository.
