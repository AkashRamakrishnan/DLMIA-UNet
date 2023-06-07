import torch
import torch.nn as nn


class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        self.encoder1 = self.conv_block(in_channels, 32)
        self.encoder2 = self.conv_block(32, 64)
        self.encoder3 = self.conv_block(64, 128)
        self.encoder4 = self.conv_block(128, 256)
        self.encoder5 = self.conv_block(256, 512)

        self.decoder1 = self.conv_block(512 + 256, 256)
        self.decoder2 = self.conv_block(256 + 128, 128)
        self.decoder3 = self.conv_block(128 + 64, 64)
        self.decoder4 = self.conv_block(64 + 32, 32)

        self.conv = nn.Conv3d(32, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)
        
        print(enc4.shape, enc5.shape)
        dec1 = self.decoder1(torch.cat([enc5, enc4], dim=1))
        dec2 = self.decoder2(torch.cat([dec1, enc3], dim=1))
        dec3 = self.decoder3(torch.cat([dec2, enc2], dim=1))
        dec4 = self.decoder4(torch.cat([dec3, enc1], dim=1))

        output = self.conv(dec4)
        return output
