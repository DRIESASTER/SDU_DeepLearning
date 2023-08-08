import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import seaborn as sns
import matplotlib.pyplot as plt


transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64,    shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

loss_values = []
val_loss_values = []



#Download the MNIST dataset
class MnistNetwork(nn.Module):

    def __init__(self):
        super(MnistNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model = MnistNetwork()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(input, target)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    loss_values = []
    accuracy_values = []


    #train the network
    for epoch in range(10):
        model.train()
        loss_values1 = []
        for data, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss_values1.append(loss.item())
            loss.backward()
            optimizer.step()
        loss_values.append(sum(loss_values1)/len(loss_values1))
        print('Epoch: %d, Loss: %.3f' % (epoch+1, sum(loss_values1)/len(loss_values1)))



        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            val_losses1 = []
            for data, targets in test_loader:
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                val_loss = criterion(outputs, targets)
                val_losses1.append(val_loss.item())
            val_loss_values.append(sum(val_losses1)/len(val_losses1))

    plt.plot(loss_values, label='Training loss')
    plt.plot(val_loss_values, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()


