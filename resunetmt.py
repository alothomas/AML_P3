import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class ResNetUNetMultiTask(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        # Load a pre-trained ResNet-50 model
        self.base_model = resnet50(pretrained=True)
        self.base_layers = list(self.base_model.children())

        # Layers from the ResNet-50
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer0_1x1 = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer1_1x1 = nn.Conv2d(256, 64, kernel_size=1, stride=1)
        self.layer2 = self.base_layers[5]
        self.layer2_1x1 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        self.layer3 = self.base_layers[6]
        self.layer3_1x1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1)
        self.layer4 = self.base_layers[7]
        self.layer4_1x1 = nn.Conv2d(2048, 512, kernel_size=1, stride=1)

        # Upsampling layers
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up3 = self.conv_block(256 + 512, 256)
        self.conv_up2 = self.conv_block(128 + 256, 128)
        self.conv_up1 = self.conv_block(128 + 64, 64)
        self.conv_up0 = self.conv_block(64 + 64, 64)
        self.conv_original_size0 = self.conv_block(3, 64)
        self.conv_original_size1 = self.conv_block(64, 64)
        self.conv_original_size2 = self.conv_block(64 + 64, 64)

        # Final convolution
        self.conv_last = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

        # Additional layers for the rotation task
        self.fc_rotation = nn.Linear(512, 4)

        # Additional layers for the jigsaw puzzle task
        self.fc_jigsaw = nn.ModuleList([nn.Linear(512, 9) for _ in range(9)])

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder part
        x_original = self.conv_original_size0(x)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        x = self.conv_last(x)
        x = torch.sigmoid(x)
        return x

    def forward_rotation(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Apply the 1x1 conv to the last layer's output
        x = self.layer4_1x1(x)

        # Adaptive average pooling and flattening
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        # Fully connected layer for rotation classification
        x = self.fc_rotation(x)
        return F.log_softmax(x, dim=1)

    def forward_jigsaw(self, x):
        bs, _, _, _ = x.shape

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Apply the 1x1 conv to the last layer's output
        x = self.layer4_1x1(x)

        # Adaptive average pooling and flattening
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        # Predictions for each jigsaw piece
        outputs = []
        for fc in self.fc_jigsaw:
            out = fc(x)
            outputs.append(F.log_softmax(out, dim=1))

        return torch.stack(outputs, dim=1).view(bs, 9, 9)



if __name__ == "__main__":
    model = ResNetUNetMultiTask(n_classes=2)
    input_tensor = torch.rand(1, 3, 224, 224)  # Example input tensor
    output = model.forward(input_tensor)  # Inpainting
    output_rotation = model.forward_rotation(input_tensor)  # Rotation
    output_jigsaw = model.forward_jigsaw(input_tensor)  # Jigsaw
