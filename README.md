# Overview
This repository provides a comprehensive comparison of training the SegFormer model from scratch versus using the Hugging Face library. The goal is to highlight the differences in implementation, training time, performance, and ease of use, along with the pros and cons of each method.

## SegFormer Overview
<div align="center">
  <img src="./resources/image.png" height="400">
</div>
<p align="center">
  Performance vs. model efficiency on ADE20K dataset.
</p>

### [Paper](https://arxiv.org/abs/2105.15203)

SegFormer is an efficient and powerful method for semantic segmentation using the Transformer architecture. It has fewer parameters than other models and can run in real-time.

## Inference Results: Training from Scratch vs. Hugging Face

<div align="center">
  <strong style="display: inline-block; margin: 0 20px;">Training from Scratch</strong> 
</div>

<div align="center"> 
  <img src="./resources/segformer_scratch_5.gif" alt="Bottom Left Video" width="600"/>
  <!-- <img src="./resources/hugginface_5.gif" alt="Bottom Right Video" width="400"/> --> 
</div>
<div align="center">
  <!-- Top GIF -->
  <img src="./resources/demo_video_4.gif" alt="Top Video" width="500"/>
</div>

<div align="center"> 
  <img src="./resources/hugginface_5.gif" alt="Bottom Right Video" width="600"/> 
</div>
<p align="center">
  <strong style="display: inline-block; margin: 0 20px;">Hugging Face</strong>
</p>

## Performance Study

![alt text](./resources/image2.png)

I fine-tuned models on the [Cityscapes dataset](https://www.cityscapes-dataset.com/). Two different pretrained models from Hugging Face were evaluated. The first model, "nvidia/segformer-b0-finetuned-ade-512-512," pretrained on the ADE20K dataset, performed poorly on Cityscapes. The second model, "nvidia/segformer-b0-finetuned-cityscapes-1024-1024," showed better results but still didn't match the performance of a model trained from scratch. The scratch-trained model, which used an ImageNet-pretrained backbone and was fine-tuned on Cityscapes, yielded the best performance.


## Training SegFormer from Scratch

Training the SegFormer model from scratch involves:

Data preprocessing: The images and labels are loaded from files. The images are then converted to tensors and normalized using specific mean and standard deviation values: mean=(0.485, 0.56, 0.406) and std=(0.229, 0.224, 0.225)

Training loop:  The CrossEntropyLoss function is used to measure the difference between the model's predictions and the actual labels. The Adam optimizer is used as optimizer.

Evaluation metrics: The mean Intersection over Union (meanIoU) is used as the evaluation metric to assess the model's performance.

Detailed steps and code are provided in the FromScratch/ directory.

## Training SegFormer Using Hugging Face

The steps taken to train the SegFormer model using the Hugging Face library were as follows:

Data handling: The SegformerFeatureExtractor API was used with the pretrained model "nvidia/segformer-b0-finetuned-cityscapes-1024-1024" to prepare the data for Hugging Face pretrained models.
Fine-tuning and evaluation: the PyTorch Lightning framework was used to fine-tune the pretrained Hugging Face model, with mean Intersection over Union (meanIoU) as the evaluation metric.
The code and instructions can be found in the HuggingFace/ directory.

## Comparison
**Implementation**

- From Scratch: Requires detailed knowledge of model architecture, data handling, and training processes.
- Using Hugging Face: Simplifies implementation by providing pre-built components and functions.

**Training Time**

- From Scratch: typically takes longer due to the need for custom implementations and optimizations. However, for this study, it was quicker because a pretrained ImageNet model was used as the backbone, resulting in a fine-tuning time of 13 hours on one Nvidia GPU.
- Using Hugging Face: it took much longer, about 24 hours to complete 32 epochs on one Nvidia GPU.

**Performance**

- From Scratch: Offers more control over model customization and optimization.
- Using Hugging Face: Provides strong baseline performance with pre-trained models but limits customization. This approach offers less  flexibility and control over certain aspects of the model and creates a dependency on external libraries.


**Ease of Use**

- From Scratch: Requires more coding effort and deep understanding of the model internal architecture.
- Using Hugging Face: provides a quicker and easier implementation. However, if you're using their tools and API for the first time, it can be challenging, with potential issues like PyTorch version mismatches and Docker image errors..

## References:

SegFormer: [arXiv Paper](https://arxiv.org/abs/2105.15203)

Hugging Face: [Hugging Face Documentation](https://huggingface.co/docs/transformers/en/model_doc/segformer)

## Acknowledgment:

I took this course that helped me understand the SegFormer architecture and coding from scratch better (https://courses.thinkautonomous.ai/view/courses/segformers-exploration/1563205-segformers-workshop/4926978-visualizing-the-attention-maps)


