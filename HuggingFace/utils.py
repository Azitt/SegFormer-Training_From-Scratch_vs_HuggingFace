import os
import cv2
import numpy as np
from collections import namedtuple

# DL library imports
import torch
from torchvision import transforms
from torch.utils.data import Dataset

#Define the Cityscapes class labels
cs_labels = namedtuple('CityscapesClass', ['name', 'train_id', 'color'])
cs_classes = [
    cs_labels('road', 0, (128, 64, 128)),
    cs_labels('sidewalk', 1, (244, 35, 232)),
    cs_labels('building', 2, (70, 70, 70)),
    cs_labels('wall', 3, (102, 102, 156)),
    cs_labels('fence', 4, (190, 153, 153)),
    cs_labels('pole', 5, (153, 153, 153)),
    cs_labels('traffic light', 6, (250, 170, 30)),
    cs_labels('traffic sign', 7, (220, 220, 0)),
    cs_labels('vegetation', 8, (107, 142, 35)),
    cs_labels('terrain', 9, (152, 251, 152)),
    cs_labels('sky', 10, (70, 130, 180)),
    cs_labels('person', 11, (220, 20, 60)),
    cs_labels('rider', 12, (255, 0, 0)),
    cs_labels('car', 13, (0, 0, 142)),
    cs_labels('truck', 14, (0, 0, 70)),
    cs_labels('bus', 15, (0, 60, 100)),
    cs_labels('train', 16, (0, 80, 100)),
    cs_labels('motorcycle', 17, (0, 0, 230)),
    cs_labels('bicycle', 18, (119, 11, 32)),
    cs_labels('ignore_class', 19, (0, 0, 0)),
]

train_id_to_color = [c.color for c in cs_classes if (c.train_id != -1 and c.train_id != 255)]
train_id_to_color = np.array(train_id_to_color)

class CityScapeDataset(Dataset):
    def __init__(self, rootDir:str, folder:str, feature_extractor, tf=None):
        """Dataset class for Cityscapes semantic segmentation data
        Args:
            rootDir (str): path to directory containing cityscapes image data
            folder (str) : 'train' or 'val' folder
            feature_extractor (SegformerFeatureExtractor): feature extractor for Segformer
        """
        self.rootDir = rootDir
        self.folder = folder
        self.feature_extractor = feature_extractor
        self.transform = tf

        # read rgb image list
        sourceImgFolder = os.path.join(self.rootDir, 'leftImg8bit', self.folder)
        self.sourceImgFiles = [os.path.join(sourceImgFolder, x) for x in sorted(os.listdir(sourceImgFolder))]

        # read label image list
        labelImgFolder = os.path.join(self.rootDir, 'gtFine', self.folder)
        self.labelImgFiles = [os.path.join(labelImgFolder, x) for x in sorted(os.listdir(labelImgFolder))]

    def __len__(self):
        return len(self.sourceImgFiles)

    def __getitem__(self, index):
        # read source image and convert to RGB
        sourceImage = cv2.imread(self.sourceImgFiles[index], -1)
        sourceImage = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2RGB)

        # read label image and convert to torch tensor
        labelImage = cv2.imread(self.labelImgFiles[index], -1)
        labelImage[labelImage == 255] = 19  # Handle void class

        # Apply feature extractor
        encoded_inputs = self.feature_extractor(sourceImage, labelImage, return_tensors="pt")

        # Remove batch dimension
        for k,v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()

        return encoded_inputs

def get_cs_datasets(rootDir,feature_extractor):
   

    # Create the initial dataset from the training folder
    data = CityScapeDataset(rootDir, folder='train', feature_extractor=feature_extractor)

    # Split the data into training and validation sets
    total_count = len(data)
    train_count = int(0.8 * total_count)
    val_count = total_count - train_count
    train_set, val_set = torch.utils.data.random_split(data, (train_count, val_count),
                                                      generator=torch.Generator().manual_seed(1))

    # Use the validation folder as the test set
    test_set = CityScapeDataset(rootDir, folder='val', feature_extractor=feature_extractor)

    return train_set, val_set, test_set  