import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#normalize transform
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


# Download the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True,
download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False,
download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
batch_size=64, shuffle=False)


# Get labels from train dataset
train_labels = train_dataset.targets

# Calculate the frequency of each digit
digit_counts = torch.bincount(train_labels, minlength=10)

# Plot the digit distribution
plt.bar(range(10), digit_counts)
plt.xlabel('Digit')
plt.ylabel('Count')
plt.title('Digit Distribution in Train Dataset')
plt.xticks(range(10))
plt.show()


# Define the model
dropout_rate = 0.25
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 13 * 13, 128)
        self.dropout = nn.Dropout(dropout_rate)
        #relu activation
        self.fc2 = nn.Linear(128, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

train_losses = []
val_losses = []

#train the model
model = CNN()
initial_filters = model.conv1.weight.data.clone()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.015, momentum=0.8) # without momentum to show learning curve. You get better performance with momentum
L2_lambda = 0.005

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    train_losses.append(loss.item())

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data)
            loss = criterion(outputs, targets)

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    val_losses.append(loss.item())

torch.save(model.state_dict(), 'model.pth')
trained_filters = model.conv1.weight.data.clone()

# Plot the training and validation losses
plt.figure(figsize=(8, 6))
plt.plot(range(num_epochs), train_losses, label="Training loss")
plt.plot(range(num_epochs), val_losses, label="Validation loss")
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.savefig('graph.png')
plt.show()


#plot conv filters
def plot_filters(filters, num_columns=8):
    num_filters = filters.size(0)
    num_rows = int(np.ceil(num_filters / num_columns))

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 15))
    for i in range(num_filters):
        row = i // num_columns
        col = i % num_columns
        ax = axes[row, col]
        ax.imshow(filters[i][0].cpu().detach(), cmap='gray')
        ax.axis('off')

    for i in range(num_filters, num_rows * num_columns):
        axes.flatten()[i].axis('off')

    plt.subplots_adjust(hspace=0.5)
    plt.show()


# Plot the initial filters
plot_filters(initial_filters, num_columns=8)

# Plot the trained filters
plot_filters(trained_filters, num_columns=8)




#plot activations for test img
# Pass the image through the first convolutional layer
# load in img.png
image = torch.from_numpy(np.array(plt.imread('imgTwo.png')))
activation = model.conv1(image)


# Visualize the activations
def visualize_activations(activation_maps, num_columns=4):
    num_maps = activation_maps.size(1)
    num_rows = int(torch.ceil(torch.tensor(num_maps).float() / num_columns))

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(12, 12))
    for i in range(num_maps):
        row = i // num_columns
        col = i % num_columns
        ax = axes[row, col]
        ax.imshow(activation_maps[0, i].cpu().detach(), cmap='viridis')
        ax.axis('off')

    for i in range(num_maps, num_rows * num_columns):
        axes.flatten()[i].axis('off')

    plt.subplots_adjust(hspace=0.5)
    plt.show()


# Visualize the activations
visualize_activations(activation)


