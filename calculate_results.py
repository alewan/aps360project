# Created by Aleksei Wan on 26.03.2020

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
from torchvision import transforms
from torchvision import datasets
import json

torch.manual_seed(1)  # set the random seed

TRAINING_RESULTS = True
USE_TRANSFER_LEARNING = True

class EmotionNet(nn.Module):
    def __init__(self):
        super(EmotionNet, self).__init__()
        self.name = "EmotionNet"
        self.conv1 = nn.Conv2d(3, 5, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 10, 3)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(10 * 18 * 18, 18 * 18)
        self.fc2 = nn.Linear(18 * 18, 32)
        self.fc3 = nn.Linear(32, 8)

    def forward(self, img):
        x = self.pool1(F.relu(self.conv1(img)))
        x = self.pool2(F.relu(self.conv2(x)))
        # print("shape", x.shape)
        x = x.view(-1, 10 * 18 * 18)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class AlexASLNet(nn.Module):
    def __init__(self):
        super(AlexASLNet, self).__init__()
        self.name = "AlexEmotionNet-3layer-640-final"
        self.fc1 = nn.Linear(256*10*10, 192)
        self.fc2 = nn.Linear(192, 128)
        self.fc3 = nn.Linear(128, 8)

    def forward(self, img):
        x = img.view(-1, 256*10*10)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_from_checkpoint(net, path):
    # load the model state from a final
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint)
    net.eval()
    return


if __name__ == "__main__":
    transform = transforms.Compose([transforms.Resize((90, 160)),  # (1080,1920) (hight, width)
                                    transforms.CenterCrop(80),
                                    transforms.ToTensor()])
    if USE_TRANSFER_LEARNING:
        net = AlexASLNet()
        load_from_checkpoint(net, 'cp_tlnn')
        test_set = datasets.DatasetFolder('./data/alex-features/test', loader=torch.load, extensions=('.tensor'))
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)
    else:
        net = EmotionNet()
        load_from_checkpoint(net, 'cp_ournn')
        testFolder = datasets.ImageFolder('./data/test', transform=transform)
        test_loader = torch.utils.data.DataLoader(testFolder, batch_size=1)

    # valFolder = datasets.ImageFolder('./data/val', transform=transform)
    # val_loader = torch.utils.data.DataLoader(valFolder, batch_size=1)


    if torch.cuda.is_available():
        net.cuda()

    preds = []
    it = 0
    print('Total Iterations:', len(test_loader))
    for imgs, labels in test_loader:
        it += 1
        print('Iteration', it)
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()
        output = net(imgs)
        preds.append((output.data.cpu().numpy().tolist()[0], labels.data.cpu().numpy().tolist()[0]) if TRAINING_RESULTS
                     else output.data.cpu().numpy().tolist()[0])

    with open('lightgbm/results.json', 'w+') as f:
        json.dump(preds, f)
