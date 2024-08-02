# basic imports
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import namedtuple

# DL library imports
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler

# For dice loss function
import segmentation_models_pytorch as smp

# for interactive widgets
import IPython.display as Disp
from ipywidgets import widgets

###################################
# FILE CONSTANTS
###################################

# Convert to torch tensor and normalize images using Imagenet values
preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.56, 0.406), std=(0.229, 0.224, 0.225))
                ])


# when using torch datasets we defined earlier, the output image
# is normalized. So we're defining an inverse transformation to 
# transform to normal RGB format
inverse_transform = transforms.Compose([
        transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
    ])


# Constants for Standard color mapping
# reference : https://github.com/bdd100k/bdd100k/blob/master/bdd100k/label/label.py

Label = namedtuple( "Label", [ "name", "train_id", "color"])
drivables = [ 
             Label("direct", 0, (219, 94, 86)),        # red
             Label("alternative", 1, (86, 211, 219)),  # cyan
             Label("background", 2, (0, 0, 0)),        # black          
            ]
train_id_to_color = [c.color for c in drivables if (c.train_id != -1 and c.train_id != 255)]
train_id_to_color = np.array(train_id_to_color)



#####################################
### ROI SELECT CLASS DEFINITION ##
#####################################
class roi_select:
    def __init__(self, im, line_color=(255, 0, 0)):
        # class variables
        self.im = im
        self.line_color = line_color
        self.selected_points = self.select_roi()

    def select_roi(self):
        # Use OpenCV's selectROI to select a rectangle
        roi = cv2.selectROI("Select ROI", self.im, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")
        return [(roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3])]

    def draw_roi_on_img(self):
        # Draw rectangle on the image
        img_copy = self.im.copy()
        cv2.rectangle(img_copy, self.selected_points[0], self.selected_points[1], self.line_color, 5)
        return img_copy

    def get_bbox_indices(self, scale_factor=None):
        pts = np.array(self.selected_points)
        if scale_factor is not None:
            pts = pts * scale_factor

        # bounding box coordinates as indices 
        # min_x, min_y is top left index
        # max_x, max_y is bottom right index
        roi_indices = pts.astype(int)
        indices = {}
        indices['min_x'], indices['min_y'] = np.min(roi_indices, axis=0)
        indices['max_x'], indices['max_y'] = np.max(roi_indices, axis=0)
        return indices

#####################################
### TORCH DATASET CLASS DEFINITION ##
#####################################


class BDD100k_dataset(Dataset):
    def __init__(self, images, labels, tf=None):
        """Dataset class for BDD100k_dataset drivable / segmentation data """
        self.images = images
        self.labels = labels
        self.tf = tf
    
    def __len__(self):
        return self.images.shape[0]
  
    def __getitem__(self, index):
        # read source image and convert to RGB, apply transform
        rgb_image = self.images[index]
        if self.tf is not None:
            rgb_image = self.tf(rgb_image)

        # read label image and convert to torch tensor
        label_image  = torch.from_numpy(self.labels[index]).long()
        return rgb_image, label_image  

###################################
# FUNCTION TO GET TORCH DATASET  #
###################################

def get_datasets(images, labels):
    data = BDD100k_dataset(images, labels, tf=preprocess)

    # split train data into train, validation and test sets
    total_count = len(data)
    train_count = int(0.7 * total_count) 
    valid_count = int(0.2 * total_count)
    test_count = total_count - train_count - valid_count
    train_set, val_set, test_set = torch.utils.data.random_split(data, 
                (train_count, valid_count, test_count), generator=torch.Generator().manual_seed(1))
    return train_set, val_set, test_set

###################################
# FUNCTION TO GET TORCH DATALOADER  #
###################################

def get_dataloaders(train_set, val_set, test_set):
    train_dataloader = DataLoader(train_set, batch_size=4,drop_last=True) # default bach_size = 8
    val_dataloader   = DataLoader(val_set, batch_size=4)
    test_dataloader  = DataLoader(test_set, batch_size=4)
    return train_dataloader, val_dataloader, test_dataloader    

###################################
# METRIC CLASS DEFINITION
###################################

class meanIoU:
    """ Class to find the mean IoU using confusion matrix approach """    
    def __init__(self, num_classes):
        self.iou_metric = 0.0
        self.num_classes = num_classes
        # placeholder for confusion matrix on entire dataset
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, y_preds, labels):
        """ Function finds the IoU for the input batch
        and add batch metrics to overall metrics """
        predicted_labels = torch.argmax(y_preds, dim=1)
        batch_confusion_matrix = self._fast_hist(labels.numpy().flatten(), predicted_labels.numpy().flatten())
        self.confusion_matrix += batch_confusion_matrix
    
    def _fast_hist(self, label_true, label_pred):
        """ Function to calculate confusion matrix on single batch """
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def compute(self):
        """ Computes overall meanIoU metric from confusion matrix data """ 
        hist = self.confusion_matrix
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        return mean_iu

    def reset(self):
        self.iou_metric = 0.0
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

###################################
# PSPNET LOSS FUNCTION DEFINITION
###################################

class pspnet_loss(nn.Module):
    def __init__(self, num_classes, aux_weight):
        super(pspnet_loss, self).__init__()
        self.aux_weight = aux_weight
        self.loss_fn = smp.losses.DiceLoss('multiclass', 
                        classes=np.arange(num_classes).tolist(), log_loss = True, smooth=1.0)
    
    def forward(self, preds, labels):
        # if input predictions is in dict format
        # calculate total loss as weighted sum of 
        # main and auxiliary losses
        if(isinstance(preds, dict) == True):
            main_loss = self.loss_fn(preds['main'], labels)
            aux_loss = self.loss_fn(preds['aux'], labels)
            loss = (1 - self.aux_weight) * main_loss + self.aux_weight * aux_loss
        else:
            loss = self.loss_fn(preds, labels)
        return loss

###################################
# POLY LR DECAY SCHEDULER DEFINITION
###################################

class polynomial_lr_decay(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    
    Reference:
        https://github.com/cmpark0126/pytorch-polynomial-lr-decay
    """
    def __init__(self, optimizer, max_decay_steps, end_learning_rate=0.0001, power=1.0):
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)
        
    def get_lr(self):
        if self.last_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [(base_lr - self.end_learning_rate) * 
                ((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
                self.end_learning_rate for base_lr in self.base_lrs]
    
    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.last_step <= self.max_decay_steps:
            decay_lrs = [(base_lr - self.end_learning_rate) * 
                         ((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
                         self.end_learning_rate for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr

###################################
# FUNCTION TO PLOT TRAINING, VALIDATION CURVES
###################################


def plot_training_results(df, model_name):
    fig, ax1 = plt.subplots(figsize=(10,4))
    ax1.set_ylabel('trainLoss', color='tab:red')
    ax1.plot(df['epoch'].values, df['trainLoss'].values, color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()  
    ax2.set_ylabel('validationLoss', color='tab:blue')
    ax2.plot(df['epoch'].values, df['validationLoss'].values, color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.suptitle(f'{model_name} Training, Validation Curves')
    plt.show()

###################################
# FUNCTION TO EVALUATE MODEL ON DATALOADER
###################################

def evaluate_model(model, dataloader, criterion, metric_class, num_classes, device):
    model.eval()
    total_loss = 0.0
    metric_object = metric_class(num_classes)

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, total=len(dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)                
            y_preds = model(inputs)

            # calculate loss
            loss = criterion(y_preds, labels)
            total_loss += loss.item()

            # update batch metric information            
            metric_object.update(y_preds.cpu().detach(), labels.cpu().detach())

    evaluation_loss = total_loss / len(dataloader)
    evaluation_metric = metric_object.compute()
    return evaluation_loss, evaluation_metric

###################################
# FUNCTION TO TRAIN, VALIDATE MODEL ON DATALOADER
###################################

def train_validate_model(model, num_epochs, model_name, criterion, optimizer, 
                         device, dataloader_train, dataloader_valid, 
                         metric_class, metric_name, num_classes, lr_scheduler = None,
                         output_path = 'code'):
    # initialize placeholders for running values
    results = []    
    min_val_loss = np.Inf
    len_train_loader = len(dataloader_train)

    # move model to device
    model.to(device)
    
    for epoch in range(num_epochs):
        print(f"Starting {epoch + 1} epoch ...")
        
        # Training
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(dataloader_train, total=len_train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device) 

            # Forward pass
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)
            train_loss += loss.item()
              
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # adjust learning rate
            if lr_scheduler is not None:
                lr_scheduler.step()
            
        # compute per batch losses, metric value
        train_loss = train_loss / len(dataloader_train)
        validation_loss, validation_metric = evaluate_model(
                        model, dataloader_valid, criterion, metric_class, num_classes, device)

        print(f'Epoch: {epoch+1}, trainLoss:{train_loss:6.5f}, validationLoss:{validation_loss:6.5f}, {metric_name}:{validation_metric: 4.2f}')
        
        # store results
        results.append({'epoch': epoch, 
                        'trainLoss': train_loss, 
                        'validationLoss': validation_loss, 
                        f'{metric_name}': validation_metric})
        
        # if validation loss has decreased, save model and reset variable
        if validation_loss <= min_val_loss:
            min_val_loss = validation_loss
        if epoch %4 == 0:    
         torch.save(model.state_dict(), f"/code/{model_name}_{epoch+1}.pt")
       


    # plot results
    print("end of training")
    torch.save(model.state_dict(), f"/code/{model_name}_end.pt")
    results = pd.DataFrame(results)
    plot_training_results(results, model_name)
    return results



