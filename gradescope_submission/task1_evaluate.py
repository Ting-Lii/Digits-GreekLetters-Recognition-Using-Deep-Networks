# Name: Ting Li; Duo Xu. 
# Date: March 24.
# Travel day used: 2 days totally, 1 day for Ting and 1 day for Duo. 

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from task1 import MyNetwork  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyNetwork().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()  # evaluation mode disables dropout randomness

# Load MNIST test set (do not shuffle)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# get first 10 images and labels
examples = [test_dataset[i] for i in range(10)]
images = torch.stack([img for img, _ in examples]).to(device)
labels = [label for _, label in examples]

# Run the model
with torch.no_grad():
    outputs = model(images)

for i in range(10):
    output_values = outputs[i].cpu().numpy()
    predicted_label = output_values.argmax()
    output_str = ", ".join([f"{val:.2f}" for val in output_values])
    print(f"Image {i}: Output: [{output_str}], Predicted: {predicted_label}, True: {labels[i]}")

# Plot first 9 prediction in 3x3 grid
plt.figure(figsize=(6, 6))
for i in range(9):
    image = images[i].cpu().squeeze()
    pred = outputs[i].argmax().item()
    plt.subplot(3, 3, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(f"Prediction: {pred}")
    plt.axis('off')
plt.tight_layout()
plt.savefig("prediction_grid.png")
plt.show()
