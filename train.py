import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from resunetmt import ResNetUNetMultiTask

torch.cuda.empty_cache()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        #y_pred = torch.sigmoid(y_pred)   #apply sigmoid to clamp between 0 and 1
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        intersection = (y_pred * y_true).sum()
        dice = (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)

        return 1 - dice


# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.masks = os.listdir(mask_dir)

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.masks[idx])
        
        image = cv2.imread(image_path)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (512, 512))

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            ])

        original_image = transform(image)
        mask = transform(mask)
        return original_image, mask


# Data Loaders
train_dataset = CustomDataset('dataset/train', 'dataset/train_gt')
val_dataset = CustomDataset('dataset/val', 'dataset/val_gt2')
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Model, Optimizer, and Loss Functions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetUNetMultiTask(n_classes=1).to(device)
model.load_state_dict(torch.load('output_dir2/model_epoch_1.pth'))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.0001)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
# dice loss
inpainting_loss_fn = DiceLoss()


# Training Function
def train_one_epoch(epoch):
    model.train()
    inpainting_loss_total = 0.0
    inpainting_loss_throughout_epoch = []

    for input_img, mask in tqdm(train_loader, desc=f'Training epoch {epoch}'):
        input_img, mask = input_img.to(device), mask.to(device)

        optimizer.zero_grad()
        inpainting_output = model.forward(input_img)
        inpainting_loss = inpainting_loss_fn(inpainting_output, mask)

        # Combine losses and backpropagate, normalize them against each other
        inpainting_loss.backward()
        optimizer.step()

        # Accumulate individual losses for logging
        inpainting_loss_total += inpainting_loss.item()
        inpainting_loss_throughout_epoch.append(inpainting_loss.item())
    
    # Save graph of loss throughout epoch
    plt.figure(figsize=(12, 8))
    plt.plot(inpainting_loss_throughout_epoch, label='Inpainting Loss')
    plt.title('Inpainting Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'st_output_dir/internalloss_graph_epoch_{epoch}.png')
    plt.close()

    return inpainting_loss_total / len(train_loader)

# Validation Function
def validate(epoch):
    model.eval()
    inpainting_loss_total = 0.0

    with torch.no_grad():
        for input_img, mask in tqdm(val_loader, desc=f'Validating epoch {epoch}'):
            input_img, mask = input_img.to(device), mask.to(device)

            inpainting_output = model.forward(input_img)
            inpainting_loss = inpainting_loss_fn(inpainting_output, mask)
            inpainting_loss_total += inpainting_loss.item()

    return inpainting_loss_total / len(val_loader)


# Plot Loss Function
def plot_loss(epoch,train_inp_losses, val_inp_losses):
    plt.figure(figsize=(12, 8))

    # Ensure the x-axis matches the number of epochs
    epochs = range(1, epoch + 2)  # +1 for zero-based indexing, +1 for inclusive range

    plt.plot(epochs, train_inp_losses, label='Train Inpainting Loss')
    plt.plot(epochs, val_inp_losses, label='Val Inpainting Loss')
    plt.title('Inpainting Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'st_output_dir/loss_graph.png')
    plt.close()


# Save Model Function
def save_model(model, path):
    torch.save(model.state_dict(), path)


train_inp_losses = []
val_inp_losses = []

# Main Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    train_inp_loss = train_one_epoch(epoch)
    scheduler.step(train_inp_loss)
    val_inp_loss = validate(epoch)
    train_inp_losses.append(train_inp_loss)
    val_inp_losses.append(val_inp_loss)

    print(f'Epoch {epoch}, Train Inpainting Loss: {train_inp_loss}')
    print(f'Epoch {epoch}, Val Inpainting Loss: {val_inp_loss}')

    save_model(model,f'st_output_dir/model_epoch_{epoch}.pth')
    plot_loss(epoch, train_inp_losses, val_inp_losses)
