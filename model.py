import torch
import torch.nn as nn
from torchvision import models

class Network(nn.Module):
    def __init__(self):
        super().__init__()        
        self.model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        self.model.fc = nn.Sequential(nn.Flatten(),
                                      nn.Linear(1792, 625),
                                      nn.ReLU(),
                                      nn.Dropout(0.3),
                                      nn.Linear(625, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, 8))
        #  add layer freezing

    def forward(self, x):   
        x = self.model(x)
        return x
        


    def save_model(self):
        
        #############################
        # Saving the model's weitghts
        # Upload 'model' as part of
        # your submission
        # Do not modify this function
        #############################
        
        torch.save(self.state_dict(), 'model.pkl')


