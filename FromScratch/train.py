from utils import meanIoU                  
from utils import train_validate_model     
from segformer import segformer_mit_b3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import segmentation_models_pytorch as smp
from timm.models.layers import drop_path, trunc_normal_
import os
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from utils import get_dataloaders
from cityScapes_utils import get_cs_datasets
import argparse

class SegformerTraining:
    def __init__(self, root_dir, target_width, target_height, n_epochs, num_classes, max_lr, model_name=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.root_dir = root_dir
        self.target_width = target_width
        self.target_height = target_height
        self.n_epochs = n_epochs
        self.num_classes = num_classes
        self.max_lr = max_lr
        self.model_name = model_name or f'segformer_{self.target_height}_{self.target_width}'
        
        self.train_set, self.val_set, self.test_set = get_cs_datasets(rootDir=self.root_dir)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = get_dataloaders(self.train_set, self.val_set, self.test_set)

        self.criterion = nn.CrossEntropyLoss(ignore_index=19)
        self.model = self._initialize_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.max_lr)
        self.scheduler = OneCycleLR(self.optimizer, max_lr=self.max_lr, epochs=self.n_epochs, steps_per_epoch=len(self.train_dataloader),
                                    pct_start=0.3, div_factor=10, anneal_strategy='cos')

    def _initialize_model(self):
        model = segformer_mit_b3(in_channels=3, num_classes=self.num_classes).to(self.device)
        model.backbone.load_state_dict(torch.load('./segformer_mit_b3_imagenet_weights.pt', map_location=self.device))
        return model

    def summary(self):
        sample_image, sample_label = self.train_set[0]
        print(f"There are {len(self.train_set)} train images, {len(self.val_set)} validation images, {len(self.test_set)} test images")
        print(f"Input shape = {sample_image.shape}, output label shape = {sample_label.shape}")

    def train(self):
        train_validate_model(self.model, self.n_epochs, self.model_name, self.criterion, self.optimizer,
                             self.device, self.train_dataloader, self.val_dataloader, meanIoU, 'meanIoU',
                             self.num_classes, lr_scheduler=self.scheduler, output_path="")

def main(args):
    trainer = SegformerTraining(root_dir=args.root_dir, target_width=args.target_width, target_height=args.target_height, 
                                n_epochs=args.n_epochs, num_classes=args.num_classes, max_lr=args.max_lr, model_name=args.model_name)
    trainer.summary()
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Segformer Model")
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory for datasets')
    parser.add_argument('--target_width', type=int, required=True, help='Target width of the images')
    parser.add_argument('--target_height', type=int, required=True, help='Target height of the images')
    parser.add_argument('--n_epochs', type=int, required=True, help='Number of epochs for training')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes for segmentation')
    parser.add_argument('--max_lr', type=float, required=True, help='Maximum learning rate')
    parser.add_argument('--model_name', type=str, default=None, help='Model name (optional)')

    args = parser.parse_args()
    main(args)
