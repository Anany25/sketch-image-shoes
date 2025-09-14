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
