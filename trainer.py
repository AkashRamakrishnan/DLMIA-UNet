import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable

import matplotlib.pyplot as plt

def train(model, train_loader, val_loader, optimizer, scheduler, device, num_classes=4, patience=3, num_epochs=20, save_path='best_model.pt', load_path=None):
    model.to(device)
    best_loss = float('inf')
    best_epoch = 0
    no_improvement = 0

    train_losses = []
    val_losses = []

    fig, ax = plt.subplots()  # Create a figure and axis object for the plot
    
    if load_path is not None:
        model.load_state_dict(torch.load(load_path))
        print('Model loaded from', load_path)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        print('** '*40)
        print('Epoch [{}/{}]'.format(epoch, num_epochs))
        print('Training')
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            data = data[:,None, :]
            target = target.type(torch.LongTensor).to(device)
            optimizer.zero_grad()
            output = model(data)
            output_dice = torch.argmax(output, dim=1)
            loss = dice_loss(output_dice, target, num_classes)
            loss = Variable(loss.data, requires_grad=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch, num_epochs, avg_loss))
        print('Validation')
        val_loss = test(model, val_loader, device, num_classes)
        val_losses.append(val_loss)
        print('Epoch [{}/{}], Validation Loss: {:.4f}'.format(epoch, num_epochs, val_loss))

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

def test(model, test_loader, device, num_classes=4):
    model.eval()
    total_loss = 0
    total_loss_2 = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
            data, target = data.to(device), target.to(device)
            data = data[:,None, :]
            target = target.type(torch.LongTensor).to(device)
            output = model(data)
            output_dice = torch.argmax(output, dim=1)
            loss = dice_loss(output_dice, target, num_classes)
            total_loss += loss.item()
            total_samples += data.size(0)

    avg_loss = total_loss / len(test_loader)
    return avg_loss

def compute_iou(pred, target, num_classes=4):
    intersection = torch.logical_and(pred, target).sum((2, 3))
    union = torch.logical_or(pred, target).sum((2, 3))
    iou = torch.zeros(num_classes)
    
    for class_idx in range(num_classes):
        class_intersection = intersection[:, class_idx]
        class_union = union[:, class_idx]
        
        class_iou = (class_intersection.float() + 1e-6) / (class_union.float() + 1e-6)
        iou[class_idx] = class_iou.mean()
    
    return iou

def dice_loss(pred_masks, true_masks, num_classes):
    epsilon = 1e-7  # Small constant to avoid division by zero
    
    dice_scores = torch.zeros(num_classes, device=pred_masks.device)
    
    for class_id in range(num_classes):
        pred_class = pred_masks == class_id
        true_class = true_masks == class_id

        intersection = (pred_class & true_class).sum()
        pred_area = pred_class.sum()
        true_area = true_class.sum()
        
        dice_score = (2.0 * intersection + epsilon) / (pred_area + true_area + epsilon)
        dice_scores[class_id] = dice_score
    
    return 1 - dice_scores.mean()


