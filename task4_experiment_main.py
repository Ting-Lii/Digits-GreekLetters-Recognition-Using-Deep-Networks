# Name: Ting Li; Duo Xu. 
# Date: March 27.
# Travel day used: 2 days totally, 1 day for Ting and 1 day for Duo. 

# This experiment file helps find the best CNN design for Fashion-MNIST, by exploring three dimensions of the CNN:
# 1. Number of convolution filters; 2. Number of fully connected (dense) nodes – how big the fully connected layer is.
# 3. Dropout rate – how much dropout to apply during training.
# The whole process is automated using Python and PyTorch, and results are saved in experiment_results.csv.

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from task4_experiment_network import ExperimentNetwork
from task4_experiment_train import train, evaluate_accuracy, save_model
import csv

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Data Loaders ===
train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./data_fashion', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
    batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./data_fashion', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
    batch_size=1000, shuffle=False)

# === Experiment Settings ===
num_filters_options = [10, 20]  
num_fc_nodes_options = [64, 128]  
dropout_rate_options = [0.3, 0.5]  

experiment_results = []
num_epochs = 5  # reduced from 10 to 5 for faster testing

total_runs = len(num_filters_options) * len(num_fc_nodes_options) * len(dropout_rate_options)
run_id = 1

# === Experiment Loop ===
for num_fc_nodes in num_fc_nodes_options:
    for dropout_rate in dropout_rate_options:
        for num_filters in num_filters_options:
            print(f"\n[Run {run_id}/{total_runs}] Filters={num_filters}, FC Nodes={num_fc_nodes}, Dropout={dropout_rate}")
            run_id += 1

            model = ExperimentNetwork(num_conv_filters=num_filters, num_fc_nodes=num_fc_nodes, dropout_rate=dropout_rate).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)  #  use Adam for faster convergence
            start_time = time()
            total_loss = 0

            for epoch in range(1, num_epochs + 1):
                epoch_loss = train(model, device, train_loader, optimizer, epoch)
                total_loss += epoch_loss

            training_time = time() - start_time
            average_loss = total_loss / num_epochs
            accuracy = evaluate_accuracy(model, device, test_loader)

            experiment_results.append({
                'num_filters': num_filters,
                'num_fc_nodes': num_fc_nodes,
                'dropout_rate': dropout_rate,
                'accuracy': accuracy,
                'training_time': training_time,
                'loss': average_loss
            })

            model_name = f'model_filters_{num_filters}_nodes_{num_fc_nodes}_dropout_{dropout_rate}.pth'
            save_model(model, './saved_models', model_name)

# === Plotting ============
plotting_keys = ['num_filters', 'num_fc_nodes', 'dropout_rate']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, key in enumerate(plotting_keys):
    axes[i].scatter([r[key] for r in experiment_results], [r['accuracy'] for r in experiment_results])
    axes[i].set_xlabel(key)
    axes[i].set_ylabel('Accuracy')
    axes[i].set_title(f'Accuracy vs {key}')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, key in enumerate(plotting_keys):
    axes[i].scatter([r[key] for r in experiment_results], [r['training_time'] for r in experiment_results])
    axes[i].set_xlabel(key)
    axes[i].set_ylabel('Training Time (s)')
    axes[i].set_title(f'Training Time vs {key}')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, key in enumerate(plotting_keys):
    axes[i].scatter([r[key] for r in experiment_results], [r['loss'] for r in experiment_results])
    axes[i].set_xlabel(key)
    axes[i].set_ylabel('Average Loss')
    axes[i].set_title(f'Average Loss vs {key}')
plt.tight_layout()
plt.show()

# === Save Results to CSV ===
with open('experiment_results.csv', 'w', newline='') as csvfile:
    fieldnames = ['num_filters', 'num_fc_nodes', 'dropout_rate', 'accuracy', 'training_time', 'loss']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for data in experiment_results:
        writer.writerow(data)

print("\nExperiment results saved to 'experiment_results.csv'")
