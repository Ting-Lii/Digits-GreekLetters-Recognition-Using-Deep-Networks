# Name: Ting Li; Duo Xu. 
# Date: March 25.
# Travel day used: 2 days totally, 1 day for Ting and 1 day for Duo. 
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision import datasets, transforms
from task1 import MyNetwork  

# load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyNetwork().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()

# access first conv layer weights
with torch.no_grad():
    conv1_weights = model.conv1.weight.cpu().clone()  
print(f"Conv1 weights shape: {conv1_weights.shape}")  

# Plot just the filters 
plt.figure(figsize=(8, 6))
for i in range(10):
    plt.subplot(3, 4, i + 1)
    plt.imshow(conv1_weights[i, 0].numpy(), cmap='viridis')
    plt.title(f"Filter {i}")
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.savefig("task2_conv1_filters.png")
plt.show()

# load the first training image (without normalization)
raw_transform = transforms.ToTensor()
raw_dataset = datasets.MNIST('./data', train=True, download=True, transform=raw_transform)
first_image_tensor, _ = raw_dataset[0]  
first_image_np = first_image_tensor.squeeze().numpy()

# Plot all filters and their outputs in a 2-column x 5-row layout, side-by-side in one figure
plt.figure(figsize=(10, 10))  

for i in range(10):
    kernel = conv1_weights[i, 0].numpy()
    filtered = cv2.filter2D(first_image_np, -1, kernel)

    # Left side: filters (2 c × 5 r)
    plt.subplot(5, 4, 2 * i + 1)
    plt.imshow(kernel, cmap='gray')
    plt.xticks([])
    plt.yticks([])

    # Right side: filtered outputs (2 c × 5 r, offset by 1 column)
    plt.subplot(5, 4, 2 * i + 2)
    plt.imshow(filtered, cmap='gray')
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.savefig("task2_filtered_outputs.png")
plt.show()
