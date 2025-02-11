import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


# Function to calculate mean and standard deviation
def calculate_mean_std(data_loader):
    mean, std, total_images = 0, 0, 0
    for images, _ in data_loader:
        batch_samples = images.size(0)  # Batch size
        images = images.view(batch_samples, images.size(1), -1)  # Reshape to (B, C, H*W)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples
    mean /= total_images
    std /= total_images
    return mean, std


# Data augmentations and transformations
def get_transforms(mean=None, std=None, img_size=64):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = val_transform
    return train_transform, val_transform, test_transform


# Loading test images manually using default_loader
def load_test_images(test_dir, transform):
    image_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir)]
    images = [transform(default_loader(img_path)) for img_path in image_paths]

    # Convert the list of images to a Tensor
    images_tensor = torch.stack(images)

    return images_tensor, image_paths


# Function to load data
def get_data_loaders(config):
    # First, calculate mean and std for normalization
    temp_transform = transforms.Compose([transforms.Resize((config.img_size, config.img_size)), transforms.ToTensor()])
    temp_dataset = datasets.ImageFolder(config.train_dir, transform=temp_transform)
    temp_loader = DataLoader(temp_dataset, batch_size=config.batch_size)
    mean, std = calculate_mean_std(temp_loader)

    # Create transformations with normalization
    train_transform, val_transform, test_transform = get_transforms(mean, std, config.img_size)

    # Load datasets
    train_dataset = datasets.ImageFolder(config.train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(config.val_dir, transform=val_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # Load test images manually using default_loader
    test_images, test_image_paths = load_test_images(config.test_dir, test_transform)

    # Use DataLoader with loaded images
    test_loader = torch.utils.data.DataLoader(
        list(zip(test_images, test_image_paths)), batch_size=1, shuffle=False
    )

    return train_loader, val_loader, test_loader


# Training loop
def train_model(model, train_loader, val_loader, config, optimizer, scheduler, device):
    train_loss_history, val_acc_history = [], []

    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = F.cross_entropy(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_loss_history.append(avg_loss)

        scheduler.step()

        model.eval()
        total_correct, total_samples = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                total_correct += (outputs.argmax(1) == y_batch).sum().item()
                total_samples += y_batch.size(0)

        val_accuracy = total_correct / total_samples
        val_acc_history.append(val_accuracy)
        print(f"Loss: {avg_loss:.4f} | Val Accuracy: {val_accuracy:.4f} | LR: {scheduler.get_last_lr()[0]}")

    return train_loss_history, val_acc_history


# Visualization function for plotting training loss and validation accuracy
def plot_metrics(train_loss_history, val_acc_history):
    plt.style.use('ggplot')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(train_loss_history, color='blue')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[1].plot(val_acc_history, color='green')
    axes[1].set_title('Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    plt.tight_layout()
    plt.show()


# Function to predict and plot the results for the test set
def predict_and_plot(model, test_loader, model_name, config, device):
    model.eval()

    num_images = len(test_loader.dataset)
    num_cols = 3
    num_rows = math.ceil(num_images / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows))
    axes = axes.flatten()  # Flatten the axes array for easier handling

    image_tensors = []
    image_paths = []

    with torch.no_grad():
        for i, (img_tensor, img_path) in enumerate(test_loader):
            img_tensor = img_tensor.to(device)

            # Get model predictions
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)  # Get probabilities
            confidence, pred_class = torch.max(probs, dim=1)  # Get the class with highest probability

            # Add images and their paths
            image_tensors.append(img_tensor.cpu())
            image_paths.append(img_path)

            # Display the image
            img = default_loader(img_path[0])  # Load the image (only one path in list)
            axes[i].imshow(img)
            axes[i].axis("off")
            axes[i].set_title(f"{config.class_names[pred_class.item()]} ({confidence.item():.2f})")

        # Remove unused axes if there are fewer images than available space
        for j in range(num_images, len(axes)):
            axes[j].axis("off")

    # Set the title for the whole figure
    fig.suptitle(f"{model_name}", fontsize=14)

    plt.tight_layout()  # Adjust layout to make it more compact
    plt.subplots_adjust(top=0.95)
    plt.show()
