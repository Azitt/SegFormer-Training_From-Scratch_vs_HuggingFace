SegFormer Model Training: From Scratch vs. Using Hugging Face
# Overview
This repository provides a comprehensive comparison of training the SegFormer model from scratch versus using the Hugging Face library. The goal is to highlight the differences in implementation, training time, performance, and ease of use, along with the pros and cons of each method.

## SegFormer Overview
<div align="center">
  <img src="./resources/image.png" height="400">
</div>
<p align="center">
  Figure 1: Performance vs. model efficiency on ADE20K dataset.
</p>

### [Paper](https://arxiv.org/abs/2105.15203)

SegFormer is an efficient and powerful method for semantic segmentation using the Transformer architecture. It has fewer parameters than other models and can run in real-time.

## Inference Results: Training from Scratch vs. Hugging Face

<!-- <p align="center">
  <strong>SORT tracking result</strong>
  <br>
  <img src="results/output_video_sort_2.gif" alt="sort tracking " width="75%">
</p>
<p align="center">
  <strong>DeepSORT tracking result</strong>
  <br>
  <img src="results/output_video_deepsort_2.gif" alt="deepsort tracking" width="75%">
</p>
<p align="center">
  <strong>StrongSORT++ tracking result</strong>
  <br>
  <img src="results/output_video_strong_2.gif" alt="strongsort tracking" width="75%">
</p> -->

<div align="center">
  <strong style="display: inline-block; margin: 0 20px;">Training from Scratch</strong>
  <strong style="display: inline-block; margin: 0 20px;">Hugging Face</strong>
</div>

<div align="center"> 
  <img src="./resources/segformer_scratch_5.gif" alt="Bottom Left Video" width="400"/>
  <img src="./resources/hugginface_5.gif" alt="Bottom Right Video" width="400"/> -->
</div>
<div align="center">
  <!-- Top GIF -->
  <img src="./resources/demo_video_4.gif" alt="Top Video" width="500"/>
</div>

<div align="center"> 
  <img src="./resources/hugginface_5.gif" alt="Bottom Right Video" width="400"/> -->
</div>

## Training SegFormer from Scratch

The implementation details for training the SegFormer model from scratch. The process involves:

Data preprocessing
Model architecture definition
Training loop
Evaluation metrics
Detailed steps and code are provided in the from_scratch/ directory.

## Training SegFormer Using Hugging Face

How to leverage the Hugging Face library to train the SegFormer model. The steps include:

Utilizing pre-trained models
Data handling with Hugging Face datasets
Fine-tuning and evaluation
All relevant code and instructions can be found in the using_huggingface/ directory.

All relevant code and instructions can be found in the using_huggingface/ directory.

## Comparison
**Implementation**

- From Scratch: Requires detailed knowledge of model architecture, data handling, and training processes.
- Using Hugging Face: Simplifies implementation by providing pre-built components and functions.

**Training Time**

- From Scratch: Typically longer due to the need for custom implementations and optimizations.
- Using Hugging Face: Generally faster due to optimized pre-built functions and pre-trained models.

**Performance**

- From Scratch: Offers more control over model customization and optimization.
- Using Hugging Face: Provides strong baseline performance with pre-trained models, though customization may be limited.

**Ease of Use**

- From Scratch: Requires significant coding effort and understanding of deep learning concepts.
- Using Hugging Face: User-friendly with extensive documentation and community support.
Pros and Cons
From Scratch
Pros:

Full control over the model and training process
Greater flexibility for customization
Deeper understanding of model internals
Cons:

Time-consuming and complex
Requires extensive coding and debugging
Higher learning curve
Using Hugging Face
Pros:

Quick and easy implementation
Access to pre-trained models and datasets
Extensive documentation and support
Cons:

Limited control over certain aspects of the model
Less flexibility for customization
Dependency on external libraries
Conclusion
Both approaches have their merits and can be chosen based on the specific requirements of the project. Training from scratch offers greater control and customization, while using Hugging Face provides a more streamlined and user-friendly experience.

References
SegFormer: arXiv Paper
Hugging Face: Hugging Face Documentation
