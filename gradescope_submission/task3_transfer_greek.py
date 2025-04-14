# Name: Ting Li; Duo Xu. 
# Date: March 26.
# Travel day used: 2 days totally, 1 day for Ting and 1 day for Duo. 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import ImageOps
import torchvision.transforms.functional as TF
from task1 import MyNetwork

# === Transform (common to both modes) ===
class GreekTransform:
    def __call__(self, x):
        x = TF.rgb_to_grayscale(x)
        x = TF.affine(x, angle=0, translate=(0, 0), scale=36/128, shear=0)
        x = TF.center_crop(x, (28, 28))
        x = TF.invert(x)
        return x

# === Load greek Dataset ==========
def load_greek_data(path, batch_size=5):
    transform = transforms.Compose([
        transforms.ToTensor(),        
        GreekTransform(),            
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.ImageFolder(path, transform=transform)
    print("Classes:", dataset.classes)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True), dataset.classes

# === Load pretrained model ===
def load_modified_model(train_mode):
    model = MyNetwork()
    model.load_state_dict(torch.load("mnist_cnn.pth", map_location="cpu"))

    # Replace last layer for Greek classification
    model.fc2 = nn.Linear(50, 3)

    # freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # For big mode, allow fc1 and fc2
    if train_mode == "big":
        for name, param in model.named_parameters():
            if "fc1" in name or "fc2" in name:
                param.requires_grad = True
    else:  # for small, only  fc2
        for name, param in model.named_parameters():
            if "fc2" in name:
                param.requires_grad = True

    return model

def train(model, dataloader, epochs=20, lr=0.01, optimizer_type="sgd"):
    model.train()

    # Collect trainable parameters
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    # Optimizer choice
    if optimizer_type == "adamw":
        optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    else:
        optimizer = optim.SGD(trainable_params, lr=lr)

    criterion = nn.CrossEntropyLoss()
    loss_history = []

    for epoch in range(epochs):
        total_loss = 0
        for images, labels in dataloader:
            output = model(images)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        loss_history.append(avg_loss)

    return loss_history

# === Plot Training Loss ===
def plot_loss(loss_history, mode):
    plt.plot(loss_history)
    plt.title(f"Greek Letter Training Loss ({mode})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    filename = f"task3_training_loss_{mode}.png"
    plt.savefig(filename)
    plt.show()

def main():
    mode = input("Choose training mode (small/big): ").strip().lower()

    if mode == "small":
        path = "greek_train_small/"
        save_path = "model_small_greek.pth"
        epochs = 20
        lr = 0.01
        optimizer_type = "sgd"
        batch_size = 5

    elif mode == "big":
        path = "greek_train_big/"
        save_path = "model_big_greek.pth"
        epochs = 90
        lr = 0.001
        optimizer_type = "adamw"
        batch_size = 5

    else:
        print("Invalid mode. Please choose 'small' or 'big'.")
        return

    dataloader, class_names = load_greek_data(path, batch_size=batch_size)
    model = load_modified_model(mode)
    loss_history = train(model, dataloader, epochs=epochs, lr=lr, optimizer_type=optimizer_type)
    plot_loss(loss_history, mode)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
