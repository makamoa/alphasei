from utils import *
from dataset import *
from models import *
from metrics import *
from loss import *
import torch 
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
import time



device = "cpu"
if torch.cuda.is_available():
	device = torch.device("cuda")   
	print("Running on: Cuda")
 
def train():
    pass

def validate():
    pass

if __name__ == "__main__":
    # python3 train.py --model mode --modelConfig modelConfigl --dsType dataset --datasetConfig datasetConfig --trainConfig trainConfig
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelType', type=str, help='model name')
    parser.add_argument('--modelConfig', type=str, help='model config json file')
    parser.add_argument('--dsType', type=str, help='dataset type: segy, npy')
    parser.add_argument('--datasetConfig', type=str, help='dataset config json file')
    parser.add_argument('--trainConfig', type=str, help='train config json file')
    args = parser.parse_args()
    
    # get the model
    model = get_model(args.modelType, args.modelConfig)
    
    # get the training configuration
    loss, epochs, opt, sch, metrics, (td, tl) = train_config(model, args.trainConfig)
    
    # get the data loaders 
    train_loader, val_loader = get_loaders(args.dsType, args.datasetConfig, td, tl)
    
    pass