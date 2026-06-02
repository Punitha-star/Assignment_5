# =========================================================
# DRIVER DROWSINESS DETECTION USING CNN
# TRAIN MODEL
# =========================================================

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms

from torchvision import datasets

from torch.utils.data import random_split

import matplotlib.pyplot as plt

# =========================================================
# IMAGE TRANSFORM
# =========================================================

transform_value = transforms.Compose([

    transforms.Grayscale(),

    transforms.Resize((28,28)),

    transforms.ToTensor(),

    transforms.Normalize((0.5,), (0.5,))

])

# =========================================================
# LOAD DATASET
# =========================================================

full_dataset = datasets.ImageFolder(

    root='dataset/train',

    transform=transform_value

)

# =========================================================
# SPLIT TRAIN / TEST
# =========================================================

train_size = int(0.8 * len(full_dataset))

test_size = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(

    full_dataset,

    [train_size, test_size]

)

# =========================================================
# DATA LOADER
# =========================================================

train_data_loader = torch.utils.data.DataLoader(

    train_dataset,

    batch_size=64,

    shuffle=True

)

test_data_loader = torch.utils.data.DataLoader(

    test_dataset,

    batch_size=64,

    shuffle=False

)

# =========================================================
# CNN MODEL
# =========================================================

class CNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(

            in_channels=1,

            out_channels=32,

            kernel_size=3,

            padding=1

        )

        self.pool = nn.MaxPool2d(

            kernel_size=2,

            stride=2

        )

        self.conv2 = nn.Conv2d(

            in_channels=32,

            out_channels=64,

            kernel_size=3,

            padding=1

        )

        self.fc1 = nn.Linear(

            7 * 7 * 64,

            128

        )

        self.fc2 = nn.Linear(

            128,

            4

        )

    def forward(self, x):

        x = self.pool(

            torch.relu(

                self.conv1(x)

            )

        )

        x = self.pool(

            torch.relu(

                self.conv2(x)

            )

        )

        x = x.view(-1, 7 * 7 * 64)

        x = torch.relu(

            self.fc1(x)

        )

        x = self.fc2(x)

        return x

# =========================================================
# CREATE MODEL
# =========================================================

model = CNN()

# =========================================================
# LOSS FUNCTION
# =========================================================

criterion = nn.CrossEntropyLoss()

# =========================================================
# OPTIMIZER
# =========================================================

optimizer = optim.Adam(

    model.parameters(),

    lr=0.01

)

# =========================================================
# TRAINING
# =========================================================

epochs = 15

train_loss = []

for epoch in range(epochs):

    running_loss = 0.0

    correct = 0

    total = 0

    for images, labels in train_data_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs,1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    average_loss = running_loss / len(train_data_loader)

    train_loss.append(average_loss)

    print(

        'Epoch =', epoch + 1,

        'Loss =', average_loss,

        'Accuracy =', accuracy

    )

# =========================================================
# SAVE MODEL
# =========================================================

torch.save(

    model.state_dict(),

    'models/drowsiness_cnn_model.pth'

)

print('Model Saved Successfully!')

# =========================================================
# LOSS GRAPH
# =========================================================

plt.plot(train_loss)

plt.title('Training Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.show()