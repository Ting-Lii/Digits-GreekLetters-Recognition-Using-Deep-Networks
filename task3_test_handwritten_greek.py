# Name: Ting Li; Duo Xu. 
# Date: March 25.
# Travel day used: 2 days totally, 1 day for Ting and 1 day for Duo. 
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from PIL import ImageOps
from task1 import MyNetwork

# === Greek Transform for Handwritten letters ===
class GreekTestTransform:
    def __call__(self, x):
        x = TF.rgb_to_grayscale(x)
        x = x.resize((128, 128))
        x = ImageOps.expand(x, border=10, fill=0)
        # affine scale to match MNIST digit scale
        x = TF.affine(x, angle=0, translate=(0, 0), scale=36 / 148, shear=0)
        # Center crop to 28x28
        x = TF.center_crop(x, (28, 28))

        return x  # return PIL Image


def load_model(model_type):
    model = MyNetwork()
    model.fc2 = nn.Linear(50, 3)  # replace final layer with 3 outputs

    if model_type == "small":
        model.load_state_dict(torch.load("model_small_greek.pth", map_location="cpu"))
    elif model_type == "big":
        model.load_state_dict(torch.load("model_big_greek.pth", map_location="cpu"))
    else:
        raise ValueError("Invalid model type")

    model.eval()
    return model

def load_test_data():
    transform = transforms.Compose([
        GreekTestTransform(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.ImageFolder("greek_test_handwritten/", transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader, dataset.classes


# === Predict and Visualize Results ===
def test_model(model, dataloader, class_names, model_type):
    plt.figure(figsize=(9, 6))
    count = 0
    for images, _ in dataloader:
        output = model(images)
        pred = torch.argmax(output, dim=1).item()
        label = class_names[pred] if pred < len(class_names) else "Unknown"

        plt.subplot(3, 3, count + 1)
        plt.imshow(images[0][0], cmap="gray")
        plt.title(f"Pred: {label}")
        plt.axis("off")
        count += 1
        if count == 9:
            break
    plt.tight_layout()
    plt.savefig(f"task3_greek_test_predictions_{model_type}.png")
    plt.show()


def main():
    model_type = input("Choose model to test (small/big): ").strip().lower()
    model = load_model(model_type)
    dataloader, class_names = load_test_data()
    print("Detected test classes:", class_names)
    test_model(model, dataloader, class_names, model_type)


if __name__ == "__main__":
    main()
