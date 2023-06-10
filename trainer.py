import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import matplotlib.pyplot as plt

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, patience=3, num_epochs=20, save_path='best_model.pt'):
    model.to(device)
    best_loss = float('inf')
    best_epoch = 0
    no_improvement = 0

    train_losses = []
    val_losses = []

    fig, ax = plt.subplots()  # Create a figure and axis object for the plot

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        print('** '*40)
        print('Epoch [{}/{}]'.format(epoch, num_epochs))
        print('Training')
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            data = data[None, :]
            target = target.type(torch.LongTensor).to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch, num_epochs, avg_loss))
        print('Validation')
        val_loss, val_accuracy = test(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        print('Epoch [{}/{}], Validation Loss: {:.4f}, Validation Accuracy: {:.2f}%'.format(epoch, num_epochs, val_loss, val_accuracy))

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print('Early stopping. No improvement in validation loss for {} epochs.'.format(no_improvement))
                break

        # Update the plot after each epoch
        ax.plot(range(1, epoch + 1), train_losses, label='Training Loss')
        ax.plot(range(1, epoch + 1), val_losses, label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        plt.savefig('loss_plot.png')  # Save the plot as an image

    print('Best model achieved at epoch {}'.format(best_epoch))

def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
            data, target = data.to(device), target.to(device)
            data = data[None, :]
            target = target.type(torch.LongTensor).to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

            pred = torch.argmax(output, dim=1)
            iou = compute_iou(pred, target)
            total_iou += iou.item()

            total_samples += data.size(0)

    avg_loss = total_loss / len(test_loader)
    avg_iou = total_iou / total_samples
    return avg_loss, avg_iou

def compute_iou(pred, target):
    intersection = torch.logical_and(pred, target).sum((1, 2, 3))
    union = torch.logical_or(pred, target).sum((1, 2, 3))
    iou = (intersection.float() + 1e-6) / (union.float() + 1e-6)
    return iou.mean()
