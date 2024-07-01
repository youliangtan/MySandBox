# Description: This file contains a simple CNN model for CIFAR-10 dataset.


"""
Commands:

python cnn_practice.py --small_vit  --lr 0.001 --weight_decay 0.05 --batch_size 1024 --dropout 0.2 --wandb --gpu_transform
 
python cnn_practice.py --resnet18 --wandb --lr 0.0005 --weight_decay 0.05 --batch_size 1024 --dropout 0.2 --gpu_transform
"""

import argparse
import numpy as np
import torch

import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import torch.nn as nn
from sklearn.metrics import f1_score

from utils import Timer

import wandb

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride

        # Check if dimensions are changing and if so, add a conv layer to match them
        if in_channels != out_channels or stride != 1:
            self.match_dimensions = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.match_dimensions = nn.Identity()

    def forward(self, x):
        residual = self.match_dimensions(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual   # combine identity
        out = self.relu(out)
        return out


class DummyCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # in_channels, out_channels, kernel_size
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.batchnorm1 = nn.BatchNorm2d(10)
        self.residuals = self.make_residual_block(2, 10, 20)

        # self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        # self.relu2 = nn.ReLU()
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.fc = nn.Linear(20*15*15, 60)  # in_features, out_features
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(60, num_classes)

        # self.softmax = nn.Softmax(dim=1) # do not use softmax if you use CrossEntropyLoss in pytorch

    def make_residual_block(self, num_residual_blocks, in_channels, out_channels):
        layers = []
        for i in range(num_residual_blocks):
            layers.append(ResidualBlock(in_channels, out_channels))
            in_channels = out_channels  # Update in_channels to match out_channels for next block
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        input size : 3 x 32 x 32
        """
        # first conv layer
        x = self.conv1(x)       # output size: 10 x 30 x 30
        x = self.relu1(x)       # output size: 10 x 30 x 30
        x = self.maxpool1(x)    # output size: 10 x 15 x 15

        x = self.batchnorm1(x)  # output size: 10 x 15 x 15
        x = self.residuals(x)   # output size: 20 x 15 x 15

        # fully connected layer
        x = torch.flatten(x, 1)  # output size: 4500 = 20 x 15 x 15
        x = self.fc(x)          # output size: 10
        x = self.relu3(x)
        x = self.fc2(x)
        # x = self.softmax(x) # do not use softmax if you use CrossEntropyLoss in pytorch
        return x

########################################################################################

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class SimpleViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_dim=192, depth=6, num_heads=8, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.transformer(x)
        x = self.norm(x)
        x = self.head(x[:, 0])
        return x

########################################################################################

class GPUTransform:
    def __init__(self, train=True, device="cuda"):
        if train:
            self.gpu_transform = torch.nn.Sequential(
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ).to(device)
        else:
            self.gpu_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        self.device = device

    def __call__(self, img):
        return self.gpu_transform(img.to(self.device))


class CIFAR10GPU(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, download=False, device="cuda"):
        self.dataset = torchvision.datasets.CIFAR10(root, train=train, download=download)
        self.transform = transform
        self.device = device

    def __getitem__(self, index):
        img, label = self.dataset[index]
        img = TF.to_tensor(img).to(self.device)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label).to(self.device)

    def __len__(self):
        return len(self.dataset)


def get_dataloader(batch_size, device):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),  # Randomly crop the image with padding of 4
        transforms.RandomHorizontalFlip(),     # Randomly flip the image horizontally
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Loading CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        # for faster data loading
        num_workers=5, pin_memory=True, prefetch_factor=4,
        persistent_workers=True, pin_memory_device=str(device)
    )

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=5, pin_memory=True, prefetch_factor=4, persistent_workers=True # for faster data loading
    )
    return trainloader, testloader, trainset, testset

########################################################################################

def main(args):
    batch_size = args.batch_size
    epoch = args.epoch

    # check use cuda or cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    # get cifar10 dataset
    # Transformations for the training and test datasets
    # this normalization is for [-1, 1] range, not [0, 1], image = (image - mean) / std
    if args.gpu_transform:
        print("Using GPU transform with multiprocessing")
        torch.multiprocessing.set_start_method('spawn')
        gpu_transform = GPUTransform(device=device)
        trainset = CIFAR10GPU(root='./data', train=True, transform=gpu_transform, download=True, device=device)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=8, prefetch_factor=4, persistent_workers=True)

        testset = CIFAR10GPU(root='./data', train=False, transform=gpu_transform, download=True, device=device)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2, persistent_workers=True)
    else:
        trainloader, testloader, trainset, testset = get_dataloader(batch_size, device)


    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    if args.resnet18:
        model = torchvision.models.resnet18(pretrained=True)

        # freeze all layers if reqires_grad is False
        # NOTE: accuracy is lower when training only the fc layer
        for param in model.parameters():
            param.requires_grad = True

        # we will only train the fc layer
        for param in model.fc.parameters():
            param.requires_grad = True

        # Option1: change the last layer to match the number of classes in our dataset
        # model.fc = nn.Linear(512, len(classes))

        # Option2: add more fc layers
        model.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(128, len(classes))
        )

    elif args.vit:
        # NOTE: only work with 224x224 images
        model = torchvision.models.vit_b_16(pretrained=True)

        # fine-tuning the entire model        
        for param in model.parameters():
            param.requires_grad = True

        model.head = nn.Linear(768, len(classes))
        
    elif args.small_vit:
        model = SimpleViT(num_classes=len(classes), dropout=args.dropout)
    else:
        model = DummyCNN(num_classes=len(classes))

    model.to(device)

    print(model)
    num_parameters = sum(p.numel() for p in model.parameters())
    print("num of params:", num_parameters)

    # trainer
    criterion = nn.CrossEntropyLoss()
    timer = Timer()

    # adam optimizer
    # NOTE: weight_decay is something like L2 regularization
    # only in vanilla SGD, it is slightly different in Adam, or adamW
    # ref: https://benihime91.github.io/blog/machinelearning/deeplearning/python3.x/tensorflow2.x/2020/10/08/adamW.html
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,   
    )

    # init wandb
    if args.wandb:
        # log args
        wandb.init(project="cnn_practice", config=vars(args))

    print("Training started...", "total data: ", len(trainloader))
    for e in range(epoch):
        running_loss = 0.0
        
        timer.tick("timer/loading_data")
        for i, data in enumerate(trainloader, 0):
            # print("loading data time: ", time.time() - start)
            # get the inputs, move to device
            inputs, labels = data[0].to(device), data[1].to(device)
            timer.tock("timer/loading_data")

            optimizer.zero_grad()  # zero the parameter gradients

            with timer.context("timer/forward_pass"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)  # calculate loss

            with timer.context("timer/back_propagation"):
                loss.backward()  # backward pass
                optimizer.step()  # update parameters

            running_loss += loss.item()
            timer.tick("timer/loading_data")

        timer.tock("timer/loading_data")

        # print running loss
        train_loss = running_loss/len(trainloader)
        print("Epoch {} - Training loss: {}".format(e+1, train_loss))

        # run test to eval, and calculate f1 score
        correct = 0
        all_preds = []
        all_labels = []
        running_loss = 0.0
        with torch.no_grad():
            for data in testloader:
                # move data to device
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)

                loss = criterion(outputs, labels)  # calculate loss
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

                # Store predictions and labels for F1 score calculation
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = correct/len(testset)
        test_loss = running_loss/len(testloader)
        f1_val = f1_score(all_labels, all_preds, average='weighted')
        print("Epoch {} - Test loss: {}".format(e+1, test_loss))
        print("Epoch {} - Test accuracy: {}".format(e+1, accuracy))
        print("Epoch {} - Test f1 score: {}".format(e+1, f1_val))
        print("-"*50)

        # log to wandb
        if args.wandb:
            wandb.log({
                "train_loss": train_loss,
                "test_loss": test_loss,
                "test_accuracy": accuracy,
                "test_f1": f1_val,
                **timer.get_average_times(),
            })

    print("Training finished.")
    torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resnet18', action='store_true',
                        help='use resnet18')
    parser.add_argument('--vit', action='store_true', help='use vision transformer')
    parser.add_argument('--small_vit', action='store_true', help='use custom transformer')
    parser.add_argument('--batch_size', type=int,
                        default=256, help='batch size')
    parser.add_argument('--epoch', type=int, default=40,
                        help='number of epochs')
    parser.add_argument('--wandb', action='store_true', help='use wandb')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--gpu_transform', action='store_true', help='use gpu transform')
    args = parser.parse_args()
    main(args)
