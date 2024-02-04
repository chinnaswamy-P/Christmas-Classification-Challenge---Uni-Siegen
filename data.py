
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torch.utils.data import Dataset
import os
from PIL import Image
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import transforms
import natsort


class ChristmasImages(Dataset):
    
    def __init__(self, path, training=True):
        super().__init__()
        self.training = training
        self.path = path
        
        #For training data
        self.transform1 = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        #For validation data
        self.transform2 = transforms.Compose([            
             transforms.Resize(256),
             transforms.ToTensor(),
             transforms.Normalize(mean= ([0.485, 0.456, 0.406]),std = ([0.229, 0.224, 0.225]))
             ])
         
        if self.training == True:
            self.dataset = ImageFolder(root=self.path.joinpath('train'),transform=self.transform1)
        else:
            self.path = path
            self.sorted_images = natsort.natsorted(os.listdir(self.path))          
            
    def __len__(self):
        if self.training:
            return len(self.dataset)
        else: 
            return len(self.sorted_images)

    def __getitem__(self, index):
        if self.training == True:
            image, label = self.dataset[index]
            return image, label
        else:            
           
            img = os.path.join(self.path,self.sorted_images[index])            
            image = self.transform2(Image.open(img).convert("RGB")) 
            return image
            
            # except Exception as e:
            #     # Handle the exception, for example, print an error message
            #     print(f"Error processing image at {image_path}: {e}")
            #     return None  # Or raise an exception or return a default image        
        raise NotImplementedError
