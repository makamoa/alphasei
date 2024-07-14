import torch
import torch.nn as nn
import os

class SimpleConvNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1) 
            # print(x.shape)
            
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.conv3(x)
        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def load_weights(self, load_path):
        if not os.path.exists(load_path):
            print('weights not found, run init')
            return self, False
        try:
            print('try to load weights')
            self.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')), strict=False)
            print('weights loaded successfully')
        except:
            print('fail to load weights')
            print('try to load pretrained from denoise weights')
            pretrained_dict = torch.load(load_path, map_location=torch.device('cpu'))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               k not in ['conv1.weight', 'conv1.bias', 'conv3.weight', 'conv3.bias']}
            model_dict.update(pretrained_dict)
            self.load_state_dict(pretrained_dict, strict=False)
            print('weights loaded successfully')
        return self, True