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
        jigsaw_image, jigsaw_label = self.prepare_jigsaw(image)
        #inpainted_image, original_image = self.prepare_inpainting(image)
        inpainted_image, original_image = cv2.resize(cv2.imread(image_path.replace('dataset', 'dataset_inp')), (512, 512)), image[:, :, 0]

        original_image = ((image - inpainted_image)[:,:,0]) # > 20 ) * 255
        original_image = ((original_image > 50) * 255).astype(np.uint8)
        inpainted_image = image


        # Convert to PyTorch tensors with transforms.ToTensor()
        rotated_image = transforms.ToTensor()(rotated_image)
        jigsaw_image = transforms.ToTensor()(jigsaw_image)
        inpainted_image = transforms.ToTensor()(inpainted_image)
        original_image = transforms.ToTensor()(original_image)
        rotation_label = torch.tensor(rotation_label)
        jigsaw_label = torch.tensor(jigsaw_label)

        return rotated_image, jigsaw_image, inpainted_image, rotation_label, jigsaw_label, original_image

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
        x, y = np.random.randint(0, image.shape[0] - 50), np.random.randint(0, image.shape[1] - 50)
        inpainted_image = image.copy()
        inpainted_image[x:x + 50, y:y + 50] = 0
        return inpainted_image, image[:, :, 0]  # Return only one channel for ground truth


# Data Loaders
train_dataset = CustomDataset('dataset/train')
val_dataset = CustomDataset('dataset/val')
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# Model, Optimizer, and Loss Functions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetUNetMultiTask(n_classes=1).to(device)
# load pretrained model
model.load_state_dict(torch.load('output_dir2/model_epoch_1.pth'))


# Training Function
def train_one_epoch(epoch):
    model.train()
    total_loss = 0.0
    rotation_loss_total = 0.0
    jigsaw_loss_total = 0.0
    inpainting_loss_total = 0.0

    rotation_loss_throughout_epoch = []
    jigsaw_loss_throughout_epoch = []
    inpainting_loss_throughout_epoch = []
    combined_loss_throughout_epoch = []

    for rotated_img, jigsaw_img, inpaint_img, rot_label, jig_label, org_img in tqdm(train_loader, desc=f'Training epoch {epoch}'):
        rotated_img, jigsaw_img = rotated_img.to(device), jigsaw_img.to(device)
        rot_label, jig_label = rot_label.to(device), jig_label.to(device)
        inpaint_img, org_img = inpaint_img.to(device), org_img.to(device)

        optimizer.zero_grad()

        # Forward pass for each task
        rotation_output = model.forward_rotation(rotated_img)
        jigsaw_output = model.forward_jigsaw(jigsaw_img)
        inpainting_output = model.forward(inpaint_img)

        # Compute loss for each task
        rotation_loss = rotation_loss_fn(rotation_output, rot_label)
        jigsaw_loss = jigsaw_loss_fn(jigsaw_output, jig_label)
        inpainting_loss = inpainting_loss_fn(inpainting_output, org_img)

        # Combine losses and backpropagate
        total_loss = rotation_loss + jigsaw_loss + inpainting_loss
        total_loss.backward()
        optimizer.step()

        # Accumulate individual losses for logging
        rotation_loss_total += rotation_loss.item()
        jigsaw_loss_total += jigsaw_loss.item()
        inpainting_loss_total += inpainting_loss.item()

        rotation_loss_throughout_epoch.append(rotation_loss.item())
        jigsaw_loss_throughout_epoch.append(jigsaw_loss.item())
        inpainting_loss_throughout_epoch.append(inpainting_loss.item())
        combined_loss_throughout_epoch.append(total_loss.item())
    
    # Save graphs for losses throughout epoch
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(range(len(rotation_loss_throughout_epoch)), rotation_loss_throughout_epoch, label='Rotation Loss')
    plt.title('Rotation Loss')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(range(len(jigsaw_loss_throughout_epoch)), jigsaw_loss_throughout_epoch, label='Jigsaw Loss')
    plt.title('Jigsaw Loss')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(range(len(inpainting_loss_throughout_epoch)), inpainting_loss_throughout_epoch, label='Inpainting Loss')
    plt.title('Inpainting Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'output_dir/internalloss_graph_epoch_{epoch}.png')
    plt.close()

    return rotation_loss_total / len(train_loader), jigsaw_loss_total / len(train_loader), inpainting_loss_total / len(train_loader)

# Validation Function
def testt(epoch):
    model.eval()
    rotation_loss_total = 0.0
    jigsaw_loss_total = 0.0
    inpainting_loss_total = 0.0

    with torch.no_grad():
        for rotated_img, jigsaw_img, inpaint_img, rot_label, jig_label, org_img in tqdm(val_loader, desc=f'Eval epoch {epoch}'):
            rotated_img, jigsaw_img = rotated_img.to(device), jigsaw_img.to(device)
            rot_label, jig_label = rot_label.to(device), jig_label.to(device)
            inpaint_img, org_img = inpaint_img.to(device), org_img.to(device)

            # Forward pass for each task
            rotation_output = model.forward_rotation(rotated_img)
            jigsaw_output = model.forward_jigsaw(jigsaw_img)
            inpainting_output = model.forward(inpaint_img)

            # Print rotation accuracy
            rotation_preds = torch.argmax(rotation_output, dim=1)
            rotation_acc = torch.sum(rotation_preds == rot_label).item() / len(rot_label)
            print('Rotation Predictions: ', rotation_preds)
            print('Rotation Labels: ', rot_label)
            print(f'Rotation Accuracy: {rotation_acc}')

            # Print jigsaw accuracy 
            jigsaw_preds = torch.argmax(jigsaw_output, dim=1)
            jigsaw_acc = torch.sum(jigsaw_preds == jig_label).item() / len(jig_label) / 9
            print('Jigsaw Predictions: ', jigsaw_preds)
            print('Jigsaw Labels: ', jig_label)
            print(f'Jigsaw Accuracy: {jigsaw_acc}')
            #print(org_img.shape)
            # Save impainting results stacked with original image to 'test_results' folder
            org_img = org_img.cpu().numpy()
            jigsaw_img = jigsaw_img.cpu().numpy()
            inpainting_output = inpainting_output.cpu().numpy()
            for i in range(len(inpainting_output)):
                inpainted_img = inpainting_output[i, 0, :, :]
                inpainted_img = np.stack((inpainted_img,)*3, axis=-1)
                inpainted_img = cv2.resize(inpainted_img, (512, 512))
                org_img1 = org_img[i,0, :, :]
                org_img1 = np.stack((org_img1,)*3, axis=-1)
                org_img1 = cv2.resize(org_img1, (512, 512))
                stacked_img = np.hstack((inpainted_img, org_img1))
                cv2.imwrite(f'test_results/epoch_{epoch}_image_{i}.png', (stacked_img*255).astype(np.uint8))

                # Also visualize jigsaw input images
                jigsaw_img1 = jigsaw_img[i, :, :, :]
                jigsaw_img1 = np.transpose(jigsaw_img1, (1, 2, 0))
                jigsaw_img1 = cv2.resize(jigsaw_img1, (512, 512))
                cv2.imwrite(f'test_results/epoch_{epoch}_jigsaw_{i}.png', (jigsaw_img1*255).astype(np.uint8))


            quit()

testt(0)