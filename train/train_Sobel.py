import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import glob
from tqdm import tqdm
import sys
from src.utils import AverageValueMeter, Dice, to_string  # Assuming utils.py has the required functions
from src.dataset import AugmentedImageDataset  # Assuming dataset.py contains your Dataset class
from src.dataset import get_complex_training_augmentation
from src.dataset import get_validation_augmentation
from src.loss import HybridLoss
from src.modelswithWT import myModel_1


def train():

    # Configuration parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kfold = 3  # Adjust for k-fold cross-validation
    epochs = 60
    batch_size = 4
    num_workers = 0
    lr = 1e-4
    pin_memory = False
    weight_decay = 1e-5
    save_path = 'G:/cobb/VT/unet/weight/'  # Directory to save the weights
    optimizer = optim.Adam
    bce_loss = HybridLoss()  # Assuming BCE loss for binary segmentation

    # Define your model, metric, and other elements
    metric = Dice()  # Using Dice metric for evaluation

    def get_train_valid_data(fold):
        # Data split based on fold number
        train_images = glob.glob(f'G:/cobb/VT/unet/data_Sobel/f0{(fold+1)%3+1}/image/*.png') + \
                      glob.glob(f'G:/cobb/VT/unet/data_Sobel/f0{(fold+2)%3+1}/image/*.png')
        train_labels = glob.glob(f'G:/cobb/VT/unet/data_Sobel/f0{(fold+1)%3+1}/label/*.png') + \
                      glob.glob(f'G:/cobb/VT/unet/data_Sobel/f0{(fold+2)%3+1}/label/*.png')

        valid_images = glob.glob(f'G:/cobb/VT/unet/data_Sobel/f0{fold%3+1}/image/*.png')
        valid_labels = glob.glob(f'G:/cobb/VT/unet/data_Sobel/f0{fold%3+1}/label/*.png')

        train_images.sort()
        train_labels.sort()
        valid_images.sort()
        valid_labels.sort()

        return (train_images, train_labels), (valid_images, valid_labels)

    k_fold_best = [0] * kfold

    for fold in range(kfold):
        # Get train and valid data for the current fold
        (train_images, train_labels), (valid_images, valid_labels) = get_train_valid_data(fold)

        # Create datasets and data loaders
        train_dataset = AugmentedImageDataset(train_images, train_labels, get_complex_training_augmentation())
        valid_dataset = AugmentedImageDataset(valid_images, valid_labels, get_validation_augmentation())
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        # Initialize model, optimizer, and metric
        seg_model = myModel_1()  # Model initialization
        opt = optimizer(seg_model.parameters(), lr=lr, weight_decay=weight_decay)
        seg_model.to(device)
        metric.to(device)

        # Define the path to save the model weights
        model_name = os.path.join(save_path, str(fold+1), 'Sobel.pth')
        max_score = 0

        # Training loop
        for epoch in range(epochs):
            print(f'Epoch: {epoch+1}/{epochs}')

            # Training phase
            seg_model.train()
            train_logs = {}
            loss_meter = AverageValueMeter()
            metric_meter = AverageValueMeter()

            with tqdm(train_loader, desc='Train', file=sys.stdout) as iterator:
                for imgs, gts in iterator:
                    imgs = imgs.to(device, dtype=torch.float)
                    gts = gts.to(device, dtype=torch.float)
                    preds = seg_model(imgs)
                    loss = bce_loss(preds, gts)

                    # Backpropagation and optimizer step
                    loss.backward()
                    opt.step()
                    opt.zero_grad()

                    # Track loss and metric
                    loss_val = loss.cpu().detach().numpy()
                    loss_meter.add(loss_val)
                    train_logs.update({'CrossEntropy': loss_meter.mean})

                    metric_val = metric(preds, gts).cpu().detach().numpy()
                    metric_meter.add(metric_val)
                    train_logs.update({metric.__name__: metric_meter.mean})

                    iterator.set_postfix_str(to_string(train_logs))

            # Validation phase
            seg_model.eval()
            val_logs = {}
            loss_meter = AverageValueMeter()
            metric_meter = AverageValueMeter()

            with tqdm(valid_loader, desc='Valid', file=sys.stdout) as iterator:
                for imgs, gts in iterator:
                    with torch.no_grad():  # Don't compute gradients during validation
                        imgs = imgs.to(device, dtype=torch.float)
                        gts = gts.to(device, dtype=torch.float)

                        preds = seg_model(imgs)
                        loss = bce_loss(preds, gts)

                    # Track validation loss and metric
                    loss_val = loss.cpu().detach().numpy()
                    loss_meter.add(loss_val)
                    val_logs.update({'CrossEntropy': loss_meter.mean})

                    metric_val = metric(preds, gts).cpu().detach().numpy()
                    metric_meter.add(metric_val)
                    val_logs.update({metric.__name__: metric_meter.mean})

                    iterator.set_postfix_str(to_string(val_logs))

            # Check if the current fold has the best validation score (Dice coefficient)
            if max_score < val_logs['Dice']:
                max_score = val_logs['Dice']
                k_fold_best[fold] = val_logs['Dice']

                # Save the model with the best validation score
                torch.save(seg_model.state_dict(), model_name)
                print('Model saved!!!')

    print('\n')
    print(f'K-Fold average Dice: {sum(k_fold_best) / len(k_fold_best):.4f}')


if __name__ == '__main__':
    train()
