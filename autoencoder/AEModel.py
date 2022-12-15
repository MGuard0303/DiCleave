from torch import nn


class SimpleAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "simple autoencoder"

        self.encoder = nn.Sequential(
            nn.BatchNorm1d(800),
            nn.Linear(800, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 800),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.reshape(-1, 800)
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)

        return encoder_out, decoder_out, x
