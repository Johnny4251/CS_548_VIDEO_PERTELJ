import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision.transforms import v2

def get_approach_names():
    return ["DeepNet", "AllKnowingNet"]

def get_approach_description(approach_name):
    if approach_name == "DeepNet":
        return "A deep spatiotemporal convolutional network. This model performs well, but can overtrain very fast."
    elif approach_name == "AllKnowingNet":
        return "This is a basic model that takes in a pretrained S3D approach and finetunes its classifying layer."

def get_data_transform(approach_name, training):
    if approach_name == "AllKnowingNet":
        base_transform = torchvision.models.video.S3D_Weights.DEFAULT.transforms()

        if training:
            data_transform = v2.Compose([
                base_transform,
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation(degrees=25)
            ])
        else: data_transform = base_transform

        return data_transform   
    else:
        return v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((256, 256))
        ])

def get_batch_size(approach_name):
    if approach_name == "AllKnowingNet": return 32
    else: return 1

def create_model(approach_name, class_cnt):

    if approach_name == "DeepNet":
        class DeepVideoNet(nn.Module):
            def __init__(self, class_cnt):
                super().__init__()
                self.feature_extract = nn.Sequential(
                    nn.Conv3d(3, 32, kernel_size=(3,3,3), padding=(1,1,1)),
                    nn.ReLU(),
                    nn.BatchNorm3d(32),
                    nn.MaxPool3d((1,2,1)),
                    
                    nn.Conv3d(32, 64, kernel_size=(3,3,3), padding=(1,1,1)),
                    nn.ReLU(),
                    nn.Conv3d(64, 64, kernel_size=(3,3,3), padding=(1,1,1)),
                    nn.ReLU(),
                    nn.BatchNorm3d(64),
                    nn.Dropout3d(p=0.3),
                    nn.AdaptiveAvgPool3d((None, 8, 8)), 
                    
                    nn.Conv3d(64, 128, kernel_size=(3,3,3), padding=(1,1,1)),
                    nn.ReLU(),
                    nn.Conv3d(128, 128, kernel_size=(3,3,3), padding=(1,1,1)),
                    nn.ReLU(),
                    nn.BatchNorm3d(128),
                    nn.Dropout3d(p=0.3),
                    nn.AdaptiveAvgPool3d((None, 8, 8)),
                )

                self.classifier_stack = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(245760, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, 512),
                    nn.ReLU(),
                    nn.Linear(512, class_cnt)
                )

            def forward(self, x):
                x = torch.transpose(x, 1, 2)
                x = self.feature_extract(x)
                logits = self.classifier_stack(x)
                return logits
                
        return DeepVideoNet(class_cnt)

    s3d = torchvision.models.video.s3d(torchvision.models.video.S3D_Weights.DEFAULT, progress=True)
    for param in s3d.parameters():
        param.requires_grad = False
    feature_cnt = s3d.classifier[1].in_channels
    s3d.classifier = nn.Sequential(
        nn.Conv3d(feature_cnt, 4096, kernel_size=1, stride=1, bias=True),
        nn.ReLU(True),
        nn.AdaptiveAvgPool3d((1,1,1)),
        nn.Flatten(),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(4096, class_cnt)
    )

    class AllKnowingNet(nn.Module):
        def __init__(self, s3d):
            super().__init__()
            self.features = s3d.features
            self.classifier = s3d.classifier

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    model = AllKnowingNet(s3d)
    return model

###############################################################################
# TRAIN ONE EPOCH
###############################################################################
def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, _, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 2 == 0:
            loss_val = loss.item()
            current = (batch + 1) * len(X)
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

###############################################################################
# TEST/EVALUATE ONE EPOCH
###############################################################################
def test_one_epoch(dataloader, model, loss_fn, data_name, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, _, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(data_name + f" Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss

def train_model(approach_name, model, device, train_dataloader, test_dataloader):
    class_cnts = np.array([61, 22, 38, 71])
    total_samples = np.sum(class_cnts)
    class_weights = class_cnts / total_samples
    class_weights = torch.Tensor(class_weights).to(device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    if approach_name == "AllKnowingNet":
        epochs = 5
    else: epochs = 1 # deepnet overtrains quick
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_one_epoch(train_dataloader, model, loss_fn, optimizer, device)
        print(f"Finished Epoch {t+1}\n-------------------------------")
        test_one_epoch(train_dataloader, model, loss_fn, "Train", device)
        test_one_epoch(test_dataloader, model, loss_fn, "Test", device)

    print("Training Complete!")
    return model
