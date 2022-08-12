#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import smdebug.pytorch as smd
from torchvision.datasets import ImageFolder
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse
import logging
import sys

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion, device):
    '''
    TODO: Complete this function that can take a model and a
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()

    running_loss = 0
    running_corrects = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * data.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader)
    total_acc = running_corrects / len(test_loader)

    logger.info(f"Test set: Average Loss: {total_loss}")
    logger.info(f"Testing Accuracy: {total_acc}")


def train(model, train_loader, valid_loader, criterion, optimizer, device, epoch):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    hook = smd.get_hook(create_if_not_exists=True)
    if hook:
        hook.register_loss(criterion)
    model.train()

    for e in range(epoch):
        running_loss = 0
        correct = 0
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            pred = model(data)  # No need to reshape data since CNNs take image inputs
            loss = criterion(pred, target)
            running_loss += loss
            loss.backward()
            optimizer.step()
            pred = pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        print(f"Train Set: Epoch {e}: Loss {running_loss / len(train_loader.dataset)}, \
         Accuracy {100 * (correct / len(train_loader.dataset))}%")
        
        # validation
        test(model, valid_loader, criterion, device)
        
        

def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 133)
    )
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_dataset_path = os.path.join(data, 'train')
    test_dataset_path = os.path.join(data, 'test')
    validation_dataset_path=os.path.join(data, 'valid')

    train_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
        ])

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
        ])

    train_data = ImageFolder(root=train_dataset_path, transform=train_transform)
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = ImageFolder(root=test_dataset_path, transform=test_transform)
    test_data_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    validation_data = ImageFolder(root=validation_dataset_path, transform=test_transform)
    validation_data_loader  = DataLoader(validation_data, batch_size=batch_size, shuffle=True) 
    
    return train_data_loader, test_data_loader, validation_data_loader


def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train_loader, test_loader, valid_loader=create_data_loaders(args.data, 64)

    model = net().to(device)

    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train(model, train_loader, valid_loader, loss_criterion, optimizer, device, epoch=args.epochs)

    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion, device)

    '''
    TODO: Save the trained model
    '''
    torch.save(model, os.path.join(args.model_dir, "hpo_model.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    args = parser.parse_args()

    main(args)
