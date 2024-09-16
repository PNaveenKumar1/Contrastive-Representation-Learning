import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    # TODO: fill in this class with the required architecture and
    # TODO: associated forward method

    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        self.z_im = z_dim
        self.conv = nn.Sequential( 
            nn.Conv2d(3,64,kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
                                  
            nn.Conv2d(64, 64, kernel_size=3, padding="same"), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
           
                                                             
            nn.Conv2d(64, 128, kernel_size=3, padding="same"), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 128, kernel_size=3, padding="same"), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            
           
            nn.Conv2d(128, 256, kernel_size=3, padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
            
            
            nn.Conv2d(256, 512, kernel_size=3, padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

        
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2), 
           )
        
        self.fc = nn.Sequential(
             nn.Linear(512 , 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, z_dim))
        
    def forward(self, x):
        
        
        x = self.conv(x)
      
        x = torch.flatten(x, 1)  
        x = self.fc(x)
        return x

    # 76.41 accuracy og VGG11

    #     self.features = nn.Sequential(
    #         nn.Conv2d(3, 64, kernel_size=3, padding=1),
    #         nn.BatchNorm2d(64),
    #         nn.ReLU(inplace=True),
            
    #         nn.Conv2d(64, 64, kernel_size=3, padding=1),
    #         nn.BatchNorm2d(64),
    #         nn.ReLU(inplace=True),
    #         nn.MaxPool2d(kernel_size=2, stride=2),
            
    #         nn.Conv2d(64, 128, kernel_size=3, padding=1),
    #         nn.BatchNorm2d(128),
    #         nn.ReLU(inplace=True),
            
    #         nn.Conv2d(128, 128, kernel_size=3, padding=1),
    #         nn.BatchNorm2d(128),
    #         nn.ReLU(inplace=True),
    #         nn.MaxPool2d(kernel_size=2, stride=2),
            
    #         nn.Conv2d(128, 256, kernel_size=3, padding=1),
    #         nn.BatchNorm2d(256),
    #         nn.ReLU(inplace=True),
            
    #         nn.Conv2d(256, 256, kernel_size=3, padding=1),
    #         nn.BatchNorm2d(256),
    #         nn.ReLU(inplace=True),
    #         nn.MaxPool2d(kernel_size=2, stride=2),
    #     )
        
    #     self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
    #     self.classifier = nn.Sequential(
    #         nn.Linear(9216, 2048),
    #         nn.ReLU(inplace=True),
    #         nn.Dropout(0.5),
    #         nn.Linear(2048, z_dim),
    #     )

    # def forward(self, x):
    #     x = self.features(x)
    #     x = self.avgpool(x)
    #     x = torch.flatten(x, 1)
    #     x = self.classifier(x)
    #     x = F.normalize(x, p=2, dim=1)
      
    #     return x
 
    #   56 Accuracy of Linet5
    #     self.conv_block = nn.Sequential(     
    #         nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
    #         nn.BatchNorm2d(6),
    #         nn.ReLU(),
    #         nn.MaxPool2d(2, 2),

    #         nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
    #         nn.BatchNorm2d(16),
    #         nn.ReLU(),
    #         nn.MaxPool2d(2, 2)
    #     )

    #     self.linear_block = nn.Sequential(
    #         nn.Linear(16 * 5 * 5, 120),
    #         nn.ReLU(),
            
    #         nn.Linear(120, 84),
    #         nn.ReLU(),
    #         nn.Linear(84, self.z_im)
    #     )
        
    # def forward(self, x):
    #     x = self.conv_block(x)
    #     x = torch.flatten(x, 1)
    #     x = self.linear_block(x)
    #     return x
    
    # raise NotImplementedError


class Classifier(nn.Module):
    # TODO: fill in this class with the required architecture and
    # TODO: associated forward method
    def __init__(self, z_dim, num_classes = 10):
        super(Classifier, self).__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        
        self.linear_block = nn.Sequential(
           nn.Linear(self.z_dim, self.num_classes),
        )
    
    def forward(self, x):
        x = self.linear_block(x)
        return x


#     raise NotImplementedError