from __future__ import print_function, division

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os
import gc
import copy
from tqdm import tqdm
from prep_gaf import run
from params import TRAINING_PARAMS
from sklearn.metrics import f1_score


#source_dir = '/home/mane/Documents/timeseries/real_pound.csv'
data_dir = 'data_GBP_USD'
#run(source_dir, data_dir)
#gc.collect()

os.makedirs(TRAINING_PARAMS['log dir'], exist_ok=True)
writer = SummaryWriter(TRAINING_PARAMS['log dir'])

plt.ion()   # interactive mode

data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            predslist = []
            labellist = []
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # print(list(labels))
                # zero the parameter gradients
                optimizer.zero_grad()
                labellist.extend(labels.tolist())
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    predslist.extend(preds.tolist())
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            print('{} F1 score: {:.4f}'.format(phase, f1_score(labellist,
                                                               predslist,
                                                               average='macro'
                                                               )))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_loss = epoch_loss
                val_acc = epoch_acc
            else:
                train_loss = epoch_loss
                train_acc = epoch_acc
        print()
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model_ft = TRAINING_PARAMS['model']
num_classes = TRAINING_PARAMS['num classes']
num_ftrs = model_ft.fc.in_features

model_ft.fc = nn.Linear(num_ftrs, num_classes)
model_ft = model_ft.to(device)

criterion = TRAINING_PARAMS['criterion']
# criterion
# Observe that all parameters are being optimized
optimizer_ft = \
    TRAINING_PARAMS['optimizer'](model_ft.parameters(),
                                 lr=TRAINING_PARAMS['learning rate'],
                                 weight_decay=TRAINING_PARAMS['weight decay'])

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler =\
    TRAINING_PARAMS['scheduler'](optimizer_ft,
                                 step_size=TRAINING_PARAMS['scheduler step size'],  # noqa
                                 gamma=TRAINING_PARAMS['scheduler gamma'])

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=TRAINING_PARAMS['num epochs'])
