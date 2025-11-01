# By Lyhin Illia
# This file contains common functions and variables used in the project


import numpy as np 
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

import matplotlib.pyplot as plt

# slightly modified from the previous homework
def show_images(samples, labels=None, n_rows=2, n_cols=5):
    samples = np.array(samples)
    labels = np.array(labels) if labels is not None else None
    plt.figure(figsize=(2 * n_cols, 2 * n_rows))
    for idx in range(n_rows * n_cols):
        plt.subplot(n_rows, n_cols, idx + 1)
        image = samples[idx].reshape(32, 32)
        plt.imshow(image, cmap='gray')
        if labels is not None:
            plt.title(f"Label: {labels[idx]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def show_image(sample, title=None):
    sample = np.array(sample).reshape(32, 32)
    plt.figure(figsize=(2, 2))
    plt.imshow(sample, cmap='gray')
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.show()


def train_one_epoch(model, loss_fn, optimizer, training_loader, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(training_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.shape[0]
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.shape[0]

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def evaluate_model(model, loss_fn, validation_loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item() * images.shape[0]
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.shape[0]

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def train_n_epochs(model, model_name, loss_fn, optimizer, training_loader, validation_loader, device, n_epochs=-1, patience=5):
    """ n_epochs = -1 means train with early stopping"""
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_accuracy = 0.0
    epochs_without_improvement = 0
    early_stopping = False
    if n_epochs == -1:
        early_stopping = True
        n_epochs = 1_000_000
    
    for epoch in range(1, n_epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, loss_fn, optimizer, training_loader, device)
        val_loss, val_accuracy = evaluate_model(model, loss_fn, validation_loader, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch}:"
              f"\t| Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
              f"\t| Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        if val_accuracy > best_val_accuracy:
            print(f"New best model '{model_name}' found at epoch {epoch} with accuracy {val_accuracy:.4f}")
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), f"saves/{model_name}.pth")
            epochs_without_improvement = 0
        elif early_stopping:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} due to no improvement in validation accuracy.")
                break

    return train_losses, train_accuracies, val_losses, val_accuracies

def plot_training_results(train_losses, train_accuracies, val_losses, val_accuracies, model_name):
    epoch_count = len(train_losses)

    if epoch_count > 25:
        xticks = np.linspace(0, epoch_count - 1, 25, dtype=int)
    else:
        xticks = np.arange(epoch_count)
    xticks_labels = np.arange(1, epoch_count + 1)[xticks]

    plt.figure(figsize=(12, 5))
    plt.suptitle(f"Training and Validation Results for {model_name}", fontsize=16)
    plt.subplot(1, 2, 1)
    
    plt.plot(train_losses, label='Train Loss', color='cornflowerblue')
    plt.plot(val_losses, label='Validation Loss', color='salmon')
    plt.xticks(ticks=xticks, labels=xticks_labels, rotation=45)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy', color='cornflowerblue')
    plt.plot(val_accuracies, label='Validation Accuracy', color='salmon')

    best_val_accuracy = max(val_accuracies)
    best_epoch = val_accuracies.index(best_val_accuracy)
    plt.scatter(best_epoch, best_val_accuracy, color='red', label=f'Best Validation Accuracy: {best_val_accuracy:.4f}')

    plt.xticks(ticks=xticks, labels=xticks_labels, rotation=45)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='lower right')
    plt.grid()

    plt.tight_layout()
    plt.show()

def train_n_plot(model, model_name, loss_fn, optimizer, training_loader, validation_loader, device, n_epochs=-1, patience=5):
    train_losses, train_accuracies, val_losses, val_accuracies = train_n_epochs(
        model, model_name, loss_fn, optimizer, training_loader, validation_loader, device, n_epochs, patience
    )
    plot_training_results(train_losses, train_accuracies, val_losses, val_accuracies, model_name)
    return train_losses, train_accuracies, val_losses, val_accuracies


