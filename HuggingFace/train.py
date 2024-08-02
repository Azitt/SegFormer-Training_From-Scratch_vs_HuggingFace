import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import SegformerFeatureExtractor
from utils import get_cs_datasets
from model import SegformerFinetuner
import argparse

class SegformerTrainer:
    def __init__(self, root_dir, max_epochs, batch_size, model_name):
        self.root_dir = root_dir
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.model_name = model_name

        # Use feature extractor from transformers
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
        self.train_dataset, self.val_dataset, self.test_dataset = get_cs_datasets(self.root_dir, self.feature_extractor)
        self.id2label = {0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence', 5: 'pole', 6: 'traffic light', 7: 'traffic sign', 8: 'vegetation', 9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car', 14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle', 19: 'ignore_class'}
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        self.segformer_finetuner = SegformerFinetuner(
            id2label=self.id2label,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            test_dataloader=self.test_dataloader,
            metrics_interval=10,
        )

        self.early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=3,
            verbose=False,
            mode="min",
        )

        self.checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss")
        self.logger = TensorBoardLogger("lightning_logs", name=self.model_name)

        self.trainer = pl.Trainer(
            callbacks=[self.early_stop_callback, self.checkpoint_callback],
            max_epochs=self.max_epochs,
            val_check_interval=len(self.train_dataloader),
            devices=1
        )

    def summary(self):
        print(f"There are {len(self.train_dataset)} train images, {len(self.val_dataset)} validation images, {len(self.test_dataset)} test images")

    def train(self):
        self.trainer.fit(self.segformer_finetuner)
        self.trainer.test(self.segformer_finetuner)
        torch.save(self.segformer_finetuner.state_dict(), f'{self.model_name}.pth')

def main(args):
    trainer = SegformerTrainer(root_dir=args.root_dir, max_epochs=args.max_epochs, batch_size=args.batch_size, model_name=args.model_name)
    trainer.summary()
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Segformer Model with PyTorch Lightning")
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory for datasets')
    parser.add_argument('--max_epochs', type=int, required=True, help='Maximum number of epochs for training')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size for training')
    parser.add_argument('--model_name', type=str, required=True, help='Model name for saving')

    args = parser.parse_args()
    main(args)
