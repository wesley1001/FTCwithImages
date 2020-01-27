import torch
from torchvision import models


TRAINING_PARAMS = {
    'num epochs': 100,
    'criterion': torch.nn.CrossEntropyLoss(),
    'model': models.resnet18(pretrained=False),
    'learning rate': 1e-5,
    'weight decay': 1e-2,
    'optimizer': torch.optim.Adam,
    'scheduler': torch.optim.lr_scheduler.StepLR,
    'scheduler step size': 4,
    'scheduler gamma': 0.1,
    'num classes': 3,
    'log dir': 'logs'
                    }


INPUT_PARAMS = {
    'series length': 15,
    'upper ratio': 0.01,
    'lower ratio': -0.01,
    'timestemp': 7,
    'step': 1
}