import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

from resunetmt import ResNetUNetMultiTask

def ssim(img1, img2, window_size=11, size_average=True):
    """
    Calculate the SSIM (Structural Similarity Index) of two images.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Size of the SSIM window
    window = torch.ones(1, 1, window_size, window_size).to('cuda')
    window = window / window.numel()

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=1)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=1) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=1) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1, img2):
        return 1 - ssim(img1, img2, window_size=self.window_size, size_average=self.size_average)

class CombinedL1SSIMLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, window_size=11, size_average=True):
        super(CombinedL1SSIMLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss(window_size, size_average)

    def forward(self, img1, img2):
        l1 = self.l1_loss(img1, img2)
        ssim = self.ssim_loss(img1, img2)
        return self.alpha * l1 + self.beta * ssim


# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        
        image = cv2.imread(image_path)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))

        rotated_image, rotation_label = self.prepare_rotation(image)
        #jigsaw_image, jigsaw_label = self.prepare_jigsaw(image)
        inpainted_image, original_image = cv2.resize(cv2.imread(image_path.replace('dataset', 'dataset_inp')), (512, 512)), image[:, :, 0]

        # ConveRT TENSORS
        rotated_image = transforms.ToTensor()(rotated_image)
        #jigsaw_image = transforms.ToTensor()(jigsaw_image)
        inpainted_image = transforms.ToTensor()(inpainted_image)
        original_image = transforms.ToTensor()(original_image)
        rotation_label = torch.tensor(rotation_label)
        #jigsaw_label = torch.tensor(jigsaw_label)

        return rotated_image, inpainted_image, rotation_label, original_image

    def prepare_rotation(self, image):
        rotations = [0, 90, 180, 270]
        rotation_label = np.random.choice(range(len(rotations)))
        if rotation_label == 0:
            return image, rotation_label
        elif rotation_label == 1:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), rotation_label
        elif rotation_label == 2:
            return cv2.rotate(image, cv2.ROTATE_180), rotation_label
        elif rotation_label == 3:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), rotation_label

    def prepare_jigsaw(self, image):
        """
        Slice image (512x512x3) into 9 pieces with varying sizes and shuffle them.
        """
        pieces = []
        sizes = [(171, 171), (171, 171), (171, 170),  # First row sizes
                (171, 171), (171, 171), (171, 170),  # Second row sizes
                (170, 171), (170, 171), (170, 170)]  # Third row sizes

        # Slice the image
        start_x = 0
        for i in range(3):
            start_y = 0
            for j in range(3):
                size = sizes[3*i+j]
                piece = image[start_x:start_x+size[0], start_y:start_y+size[1]]
                start_y += size[1]

                # Pad the smaller pieces
                if piece.shape[0] != 171 or piece.shape[1] != 171:
                    piece = np.pad(piece, ((0, 171 - piece.shape[0]), (0, 171 - piece.shape[1]), (0, 0)), mode='constant')

                pieces.append(piece)
            start_x += sizes[3*i][0]

        # Shuffle the pieces
        shuffled_indices = np.random.permutation(9)
        shuffled_pieces = [pieces[i] for i in shuffled_indices]
        jigsaw_image = np.vstack([np.hstack(shuffled_pieces[i*3:(i+1)*3]) for i in range(3)])

        # Create jigsaw label
        jigsaw_label = np.array(shuffled_indices)
        return jigsaw_image, jigsaw_label


    def prepare_inpainting(self, image):
        x, y = np.random.randint(0, image.shape[0] - 100), np.random.randint(0, image.shape[1] - 100)
        inpainted_image = image.copy()
        inpainted_image[x:x + 100, y:y + 100] = 0
        return inpainted_image, image[:, :, 0]  # Return only one channel for ground truth


# Data Loaders
train_dataset = CustomDataset('dataset/train')
val_dataset = CustomDataset('dataset/val')
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# Model, Optimizer, and Loss Functions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetUNetMultiTask(n_classes=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)
rotation_loss_fn = nn.CrossEntropyLoss()
#jigsaw_loss_fn = nn.CrossEntropyLoss()
inpainting_loss_fn = CombinedL1SSIMLoss(alpha=0.7, beta=0.3) # nn.MSELoss() # MSE caused blurry results


# Training Function
def train_one_epoch(epoch):
    model.train()
    total_loss = 0.0
    rotation_loss_total = 0.0
    jigsaw_loss_total = 0.0
    inpainting_loss_total = 0.0

    rotation_loss_throughout_epoch = []
    #jigsaw_loss_throughout_epoch = []
    inpainting_loss_throughout_epoch = []
    combined_loss_throughout_epoch = []

    for rotated_img, inpaint_img, rot_label, org_img in tqdm(train_loader, desc=f'Training epoch {epoch}'):
        rotated_img, rot_label = rotated_img.to(device), rot_label.to(device)
        inpaint_img, org_img = inpaint_img.to(device), org_img.to(device)

        optimizer.zero_grad()

        # Forward pass for each task
        rotation_output = model.forward_rotation(rotated_img)
        #jigsaw_output = model.forward_jigsaw(jigsaw_img)
        inpainting_output = model.forward(inpaint_img)

        # Compute loss for each task
        rotation_loss = rotation_loss_fn(rotation_output, rot_label)
        #jigsaw_loss = jigsaw_loss_fn(jigsaw_output, jig_label)
        inpainting_loss = inpainting_loss_fn(inpainting_output, org_img)

        # Combine losses and backpropagate, normalize them against each other
        #total_loss = rotation_loss + inpainting_loss * 10
        total_loss = inpainting_loss
        total_loss.backward()
        optimizer.step()

        # Accumulate individual losses for logging
        rotation_loss_total += rotation_loss.item()
        #jigsaw_loss_total += jigsaw_loss.item()
        inpainting_loss_total += inpainting_loss.item()

        rotation_loss_throughout_epoch.append(rotation_loss.item())
        #jigsaw_loss_throughout_epoch.append(jigsaw_loss.item())
        inpainting_loss_throughout_epoch.append(inpainting_loss.item())
        combined_loss_throughout_epoch.append(total_loss.item())
    
    # Save graphs for losses throughout epoch
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(range(len(rotation_loss_throughout_epoch)), rotation_loss_throughout_epoch, label='Rotation Loss')
    plt.title('Rotation Loss')
    plt.legend()

    """plt.subplot(3, 1, 2)
    plt.plot(range(len(jigsaw_loss_throughout_epoch)), jigsaw_loss_throughout_epoch, label='Jigsaw Loss')
    plt.title('Jigsaw Loss')
    plt.legend()"""

    plt.subplot(3, 1, 3)
    plt.plot(range(len(inpainting_loss_throughout_epoch)), inpainting_loss_throughout_epoch, label='Inpainting Loss')
    plt.title('Inpainting Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'output_dir2/internalloss_graph_epoch_{epoch}.png')
    plt.close()

    return rotation_loss_total / len(train_loader), inpainting_loss_total / len(train_loader)

# Validation Function
def validate(epoch):
    model.eval()
    rotation_loss_total = 0.0
    jigsaw_loss_total = 0.0
    inpainting_loss_total = 0.0

    with torch.no_grad():
        for rotated_img, inpaint_img, rot_label, org_img in tqdm(val_loader, desc=f'Eval epoch {epoch}'):
            rotated_img, rot_label = rotated_img.to(device), rot_label.to(device)
            inpaint_img, org_img = inpaint_img.to(device), org_img.to(device)

            # Forward pass for each task
            rotation_output = model.forward_rotation(rotated_img)
            #jigsaw_output = model.forward_jigsaw(jigsaw_img)
            inpainting_output = model.forward(inpaint_img)

            # Compute loss for each task
            rotation_loss = rotation_loss_fn(rotation_output, rot_label)
            #jigsaw_loss = jigsaw_loss_fn(jigsaw_output, jig_label)
            inpainting_loss = inpainting_loss_fn(inpainting_output, org_img)

            # Accumulate individual losses for logging
            rotation_loss_total += rotation_loss.item()
            #jigsaw_loss_total += jigsaw_loss.item()
            inpainting_loss_total += inpainting_loss.item()

    return rotation_loss_total / len(val_loader), inpainting_loss_total / len(val_loader)


# Plot Loss Function
def plot_loss(epoch, train_rot_losses, train_inp_losses, val_rot_losses, val_inp_losses):
    plt.figure(figsize=(12, 8))

    # Ensure the x-axis matches the number of epochs
    epochs = range(1, epoch + 2)  # +1 for zero-based indexing, +1 for inclusive range

    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_rot_losses, label='Train Rotation Loss')
    plt.plot(epochs, val_rot_losses, label='Val Rotation Loss')
    plt.title('Rotation Loss')
    plt.legend()

    """plt.subplot(3, 1, 2)
    plt.plot(epochs, train_jig_losses, label='Train Jigsaw Loss')
    plt.plot(epochs, val_jig_losses, label='Val Jigsaw Loss')
    plt.title('Jigsaw Loss')
    plt.legend()"""

    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_inp_losses, label='Train Inpainting Loss')
    plt.plot(epochs, val_inp_losses, label='Val Inpainting Loss')
    plt.title('Inpainting Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'output_dir2/loss_graph.png')
    plt.close()


# Save Model Function
def save_model(model, path):
    torch.save(model.state_dict(), path)

train_rot_losses = []
#train_jig_losses = []
train_inp_losses = []
val_rot_losses = []
#val_jig_losses = []
val_inp_losses = []

# Main Training Loop
num_epochs = 20
for epoch in range(num_epochs):
    train_rot_loss, train_inp_loss = train_one_epoch(epoch)
    scheduler.step(train_inp_loss)
    val_rot_loss, val_inp_loss = validate(epoch)

    train_rot_losses.append(train_rot_loss)
    #train_jig_losses.append(train_jig_loss)
    train_inp_losses.append(train_inp_loss)
    val_rot_losses.append(val_rot_loss)
    #val_jig_losses.append(val_jig_loss)
    val_inp_losses.append(val_inp_loss)

    print(f'Epoch {epoch}, Train Rotation Loss: {train_rot_loss},  Train Inpainting Loss: {train_inp_loss}')
    print(f'Epoch {epoch}, Val Rotation Loss: {val_rot_loss},  Val Inpainting Loss: {val_inp_loss}')

    save_model(model,f'output_dir2/model_epoch_{epoch}.pth')
    plot_loss(epoch, train_rot_losses, train_inp_losses, val_rot_losses, val_inp_losses)
