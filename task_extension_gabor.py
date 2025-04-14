# Name: Ting Li; Duo Xu
# Travel day used: 2 days totally, 1 day for Ting and 1 day for Duo. 
# This file uses fixed Gabor filter replaced the first CNN layer, and we analyse the accuracy.
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from task4_experiment_network import ExperimentNetwork
from task4_experiment_train import train, evaluate_accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Fashion MNIST ===
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./data_fashion', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./data_fashion', train=False, download=True, transform=transform),
    batch_size=1000, shuffle=False
)

# === Train with Gabor Filter Layer ===
model = ExperimentNetwork(num_conv_filters=10, num_fc_nodes=64, dropout_rate=0.5, use_gabor=True).to(device)
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, momentum=0.5)

print("Training Gabor-based model...")
for epoch in range(1, 10 + 1):
    train(model, device, train_loader, optimizer, epoch)

# === Evaluate ===
accuracy = evaluate_accuracy(model, device, test_loader)
print(f"Test Accuracy using Gabor filters: {accuracy:.4f}")
