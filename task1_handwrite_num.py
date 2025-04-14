# Name: Ting Li; Duo Xu. 
# Date: March 24.
# Travel day used: 2 days totally, 1 day for Ting and 1 day for Duo. 
import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from task1 import MyNetwork  

# Define the same transform used on MNIST (grayscale + normalize)
transform = transforms.Compose([
    transforms.Grayscale(),                    
    transforms.Resize((28, 28)),                
    transforms.ToTensor(),                      
    transforms.Lambda(lambda x: 1.0 - x),       
    transforms.Normalize((0.1307,), (0.3081,))  
])

def load_handwritten_images(folder_path):
    image_tensors = []
    original_images = []
    filenames = sorted(os.listdir(folder_path))  # 0.png to 9.png
    for fname in filenames:
        if fname.endswith(".png"):
            img_path = os.path.join(folder_path, fname)
            image = Image.open(img_path).convert("L")  
            original_images.append(image.copy())      
            tensor = transform(image)
            image_tensors.append(tensor)
    return torch.stack(image_tensors), original_images, filenames

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained model
    model = MyNetwork().to(device)
    model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
    model.eval()

    # Load handwritten image
    image_folder = "handwritten"
    images_tensor, originals, filenames = load_handwritten_images(image_folder)
    images_tensor = images_tensor.to(device)

    # Run predictions
    with torch.no_grad():
        outputs = model(images_tensor)
        predictions = torch.argmax(outputs, dim=1)

    for i in range(len(filenames)):
        print(f"{filenames[i]} -> Prediction: {predictions[i].item()}")

    # plot images with predictions
    plt.figure(figsize=(10, 4))
    for i in range(len(originals)):
        plt.subplot(2, 5, i+1)
        plt.imshow(originals[i], cmap='gray')
        plt.title(f"Pred: {predictions[i].item()}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("handwritten_predictions.png")
    plt.show()

if __name__ == "__main__":
    main()
