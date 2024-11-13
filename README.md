Synthetic OCT-A Generation using Fundus Images and GANs
Overview
This repository implements a Conditional Generative Adversarial Network (cGAN) to generate synthetic Optical Coherence Tomography Angiography (OCT-A) images from fundus images. The main goal is to automate and improve the extraction of vascular maps for retinal analysis, especially in scenarios where OCT-A imaging is not feasible.

Background
OCT-A images allow precise visualization of retinal vasculature, which is useful for studying various retinal diseases and biomarkers. However, OCT-A imaging devices are expensive, less portable, and limited in field of view (FOV). This project leverages fundus images, which are more accessible and inexpensive, to create synthetic OCT-A images, providing a practical alternative for vascular analysis.

The model is trained using a dataset of aligned fundus and OCT-A images and uses a custom cGAN to map fundus images to OCT-A representations.

Methods and Pipeline
The code comprises several key modules and methods to accomplish the synthesis task:

1. Preprocessing
Preprocessing steps include:

Masking noisy pixels in the background.
Normalizing intensities between -1 and 1.
Aligning OCT-A images with corresponding fundus images to match FOV and vascular patterns.
2. Model Architecture
The model utilizes a U-Net-based generator and a CNN-based discriminator:

Generator: A U-Net with skip connections to preserve spatial information while mapping the fundus image to the OCT-A representation.
Discriminator: A CNN that distinguishes between real and synthetic OCT-A images, enhancing the generator's realism.
3. Training
Losses: Adversarial loss and mean absolute error (MSE) are used to train the generator.
Optimization: The Adam optimizer is used with a learning rate of 2e-4 for both networks.
Data Augmentation: Reflection and rotation augmentations prevent overfitting and improve generalizability.
4. Evaluation
The model is evaluated based on vessel density correlation in macular and optic disc regions, precision-recall analysis, and visual comparisons against ground truth OCT-A and vessel segmentation models such as SA-UNet and IterNet.

Requirements
Python 3.x
TensorFlow
NumPy
SciPy
Scikit-Image
Matplotlib
Pandas
Usage
Clone the repository:

bash
Copy code
git clone https://github.com/your_username/your_repo_name.git
cd your_repo_name
Install the dependencies:

bash
Copy code
pip install -r requirements.txt
To train the model:

bash
Copy code
python main.py --case_range "1_25" --gpu_id 0
Evaluation and Visualization: The trained model evaluates OCT-A generation quality on test cases, providing output images of synthetic OCT-A compared with the original OCT-A.

Results
Vessel Density: Synthetic OCT-A closely matches ground truth OCT-A in terms of vessel density, showing a significant correlation for both macular and optic disc regions.
Segmentation Accuracy: The model is comparable with IterNet and SA-UNet in terms of retinal vessel representation but does not rely on manual vessel delineation.
Dataset
The dataset contains fundus and OCT-A image pairs for various retinal regions. The images were aligned and preprocessed to ensure spatial correspondence between modalities.

References
For a detailed understanding of the methodology and experiments, refer to our paper: "Synthetic OCT-A blood vessel maps using fundus images and generative adversarial networks", available at DOI link.

Acknowledgements
This work was supported by the Translational Research Institute for Space Health through NASA Cooperative Agreement NNX16AO69A and other funding sources. Special thanks to the McWilliams School of Biomedical Informatics at UTHealth for their support and resources.