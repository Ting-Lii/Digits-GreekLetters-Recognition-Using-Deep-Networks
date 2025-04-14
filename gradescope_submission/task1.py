# Name: Ting Li; Duo Xu. 
# Date: March 22.
# Travel day used: 
# Below code adapted from Project 5: Recognition using Deep Networks which taught by Professor Bruce.
# Task 1A: Build, train, evaluate, and save a digit recognition model on MNIST using PyTorch

# import statements
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# class definitions
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    # computes a forward pass for the network
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Plot the first six images from the test set
def plot_first_six_images(test_dataset):
    fig, axes = plt.subplots(2, 3)
    for i in range(6):
        image, label = test_dataset[i]
        ax = axes[i // 3, i % 3]
        ax.imshow(image.squeeze(), cmap='gray')
        ax.set_title(f"Label: {label}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("first_six_test_images.png")
    plt.show()

def train_network(model, train_loader, test_loader, device, epochs=5, lr=0.01):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    train_acc_list, test_acc_list = [], []
    train_losses, test_losses, batches_seen = [], [], []

    seen_examples = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            # Track training loss per batch
            seen_examples += len(data)
            train_losses.append(loss.item())
            batches_seen.append(seen_examples)

            if batch_idx % 10 == 0:
                print(f'Train Epoch: {epoch} [{seen_examples}/{len(train_loader.dataset) * epochs}] '
                      f'Loss: {loss.item():.6f}')

        # Evaluate accuracy and test loss after each epoch
        train_acc = calculate_accuracy(train_loader, model, device)
        test_acc = calculate_accuracy(test_loader, model, device)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        test_loss = evaluate_test_loss(model, test_loader, device)
        test_losses.append((seen_examples, test_loss))

        print(f"Epoch {epoch}: Train Acc = {train_acc:.2f}%, Test Acc = {test_acc:.2f}%, Test Loss = {test_loss:.4f}")

    # plot Accuracy
    plt.figure()
    plt.plot(range(1, epochs + 1), train_acc_list, label='Train Accuracy')
    plt.plot(range(1, epochs + 1), test_acc_list, label='Test Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Test Accuracy")
    plt.legend()
    plt.savefig("accuracy_plot.png")
    plt.show()

    # Plot Loss (train + test)
    plt.figure()
    plt.plot(batches_seen, train_losses, label='Train Loss', color='blue')
    test_xs = [x[0] for x in test_losses]
    test_ys = [x[1] for x in test_losses]
    plt.scatter(test_xs, test_ys, color='red', label='Test Loss')
    plt.xlabel("Number of Training Examples Seen")
    plt.ylabel("Negative Log Likelihood Loss")
    plt.title("Negative Log Likelihood Loss over Number of Training Examples Seen")
    plt.legend()
    plt.savefig("plot-trainingError.png")
    plt.show()

    torch.save(model.state_dict(), "mnist_cnn.pth")


# calculate accuracy
def calculate_accuracy(loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# test loss (to plot per epoch, task1.c)
def evaluate_test_loss(model, test_loader, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
    test_loss /= len(test_loader.dataset)
    return test_loss


# main function (handles full training flow)
def main(argv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Plot first six test 
    plot_first_six_images(test_dataset)

    # initialize and train model
    model = MyNetwork().to(device)
    train_network(model, train_loader, test_loader, device)

if __name__ == "__main__":
    main(sys.argv)
