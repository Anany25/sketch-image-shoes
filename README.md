# Sketch to Photo Translation for Shoes

This project implements a Generative Adversarial Network (GAN) to translate sketches of shoes into realistic photos. The model is based on the pix2pix architecture, which uses a conditional GAN to learn the mapping from an input image to an output image.

## Description

The core of this project is an image-to-image translation model that takes a shoe sketch as input and generates a corresponding photorealistic image. This is achieved using a cGAN (conditional Generative Adversarial Network) with a U-Net based generator and a PatchGAN discriminator. The model is trained on the "edges2shoes" dataset.

## Dependencies

The following Python libraries are required to run the project:

* os
* numpy
* matplotlib
* Pillow (PIL)
* torch
* torchvision

You can install the necessary packages using pip:
```bash
pip install numpy matplotlib pillow torch torchvision
```

## Dataset

This project uses the **edges2shoes** dataset, which contains approximately 50,000 images of shoes and their corresponding edge maps.

You can download the dataset from the following link:
[http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2shoes.tar.gz](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2shoes.tar.gz)

Once downloaded, extract the dataset and organize it into the following directory structure:

```
edges2shoes/
├── train/
│   ├── ... (training images)
└── val/
    ├── ... (validation images)
```

## Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Anany25/sketch-image-shoes/](https://github.com/Anany25/sketch-image-shoes/)
    cd sketch-image-shoes
    ```

2.  **Download and prepare the dataset** as described in the "Dataset" section.

3.  **Run the Jupyter Notebook:**
    Open and run the `sketch-photo(shoe).ipynb` notebook. The notebook will:
    * Load and preprocess the dataset.
    * Define the Generator (U-Net) and Discriminator (PatchGAN) models.
    * Train the model for 100 epochs (you can adjust the hyperparameters in the notebook).
    * Display the generated images alongside the input sketches and real photos to visualize the results.

## Model Architecture

* **Generator:** The generator is based on a U-Net architecture. It takes a sketch as input and outputs a generated photo. The U-Net architecture includes skip connections between the encoder and decoder, which helps the generator to produce more detailed and realistic images.
* **Discriminator:** The discriminator is a PatchGAN, which classifies patches of an image as real or fake, rather than the entire image. This encourages the generator to produce high-frequency details and sharper images.

## Training Details

* **Loss Functions:**
    * **Adversarial Loss:** BCEWithLogitsLoss is used to train the generator and discriminator.
    * **Pixel-wise Loss:** L1Loss is used to encourage the generator to produce images that are structurally similar to the ground truth.
* **Optimizer:** The Adam optimizer is used for both the generator and discriminator.
* **Hyperparameters:**
    * Batch Size: 4
    * Image Size: 256x256
    * Learning Rate: 0.0002
    * Epochs: 100

## Results

After training, the model is able to generate realistic shoe photos from input sketches. The notebook includes a visualization section that shows a comparison between the input sketches, the generated photos, and the real photos.

## Acknowledgments

This project is based on the work of Isola et al. in their paper "[Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)" (pix2pix). The "edges2shoes" dataset is provided by the authors of the pix2pix paper.
