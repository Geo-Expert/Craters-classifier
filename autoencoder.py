import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self, bottleneck_size):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=5, stride=1, padding=2),  # Same padding
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
            nn.Linear(128 * 100 * 100, bottleneck_size)
        )
        # Decoder
        self.decoder_fc = nn.Linear(bottleneck_size, 128 * 100 * 100)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.Upsample(scale_factor=1),  # Adjust scale_factor if upsampling needed
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Output range [0, 1]
        )
    def forward(self, x):
        # Encode
        x = self.encoder(x)

        # Decode
        x = self.decoder_fc(x)
        x = x.view(-1, 128, 100, 100)  # Reshape to 4D for ConvTranspose
        x = self.decoder(x)
        return x
