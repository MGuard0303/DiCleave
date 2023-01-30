import torch
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "autoencoder"

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


class CNNModel(nn.Module):
    def __init__(self, task, name="model"):
        super().__init__()

        self.task = task
        self.name = name

        self.identity = nn.Sequential()

        self.convs = nn.ModuleDict({
            "same_1": nn.Conv1d(in_channels=13, out_channels=13, kernel_size=3, padding=1),
            "blk_1_1": nn.Conv1d(in_channels=13, out_channels=16, kernel_size=3, padding=2),
            "blk_1_2": nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),

            "same_2": nn.Conv1d(in_channels=2, out_channels=2, kernel_size=3, padding=1),
            "blk_2_1": nn.Conv1d(in_channels=2, out_channels=8, kernel_size=3, padding=1),
            "blk_2_2": nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            "blk_2_3": nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            "blk_2_4": nn.Conv1d(in_channels=32, out_channel=64, kernel_size=3, padding=1)
        })

        self.bns = nn.ModuleDict({
            "same_1": nn.BatchNorm1d(num_features=13),
            "blk_1_1": nn.BatchNorm1d(num_features=16),
            "blk_1_2": nn.BatchNorm1d(num_features=32),

            "same_2": nn.BatchNorm1d(num_features=2),
            "blk_2_1": nn.BatchNorm1d(num_features=8),
            "blk_2_2": nn.BatchNorm1d(num_features=16),
            "blk_2_3": nn.BatchNorm1d(num_features=32),
            "blk_2_4": nn.BatchNorm1d(num_features=64)
        })

        self.pool = nn.MaxPool1d(2)

        self.relu = nn.LeakyReLU()

        self.fc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.LeakyReLU()
        )

        if self.task == "binary":

            self.fc2 = nn.Sequential(
                nn.Linear(256, 128),
                nn.LeakyReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.LeakyReLU()
            )

        elif self.task == "multi":
            self.fc2 = nn.Sequential(
                nn.Linear(512, 256),
                nn.LeakyReLU(),
                nn.Dropout(),
                nn.Linear(256, 128),
                nn.LeakyReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.LeakyReLU()
            )

        self.bi_unit = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.mul_unit = nn.Sequential(
            nn.Linear(32, 3),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, embed):
        # Forward propagate in CU1 of module_1
        x_identity = self.identity(x)

        for i in range(3):
            x = self.convs["same_1"](x)
            x = self.bns["same_1"](x)
            x = self.relu(x)
            x += x_identity

        # Forward propagate in CU2 of module_1
        x = self.convs["blk_1_1"](x)
        x = self.bns["blk_1_1"](x)
        x = self.relu(x)
        x = self.convs["blk_1_2"](x)
        x = self.bns["blk_1_2"](x)
        x = self.relu(x)

        # Forward propagate in FC layers of module_1
        x = torch.reshape(x, (x.size(0), -1))
        x = self.fc1(x)

        # Concatenate sequence vector with secondary structure embedding
        x = torch.unsqueeze(x, dim=1)
        embed = torch.unsqueeze(embed, dim=1)
        x_embed = torch.cat((x, embed), dim=1)

        # Forward propagate in CU1 of module_2
        x_embed_identity = self.identity(x_embed)

        for i in range(3):
            x_embed = self.convs["same_2"](x_embed)
            x_embed = self.bns["same_2"](x_embed)
            x_embed = self.relu(x_embed)
            x_embed += x_embed_identity

        # Forward propagate in CU2 of module_2
        if self.task == "binary":
            x_embed = self.convs["blk_2_1"](x_embed)
            x_embed = self.bns["blk_2_1"](x_embed)
            x_embed = self.pool(x_embed)
            x_embed = self.relu(x_embed)

            x_embed = self.convs["blk_2_2"](x_embed)
            x_embed = self.bns["blk_2_2"](x_embed)
            x_embed = self.pool(x_embed)
            x_embed = self.relu(x_embed)

        elif self.task == "multi":
            x_embed = self.convs["blk_2_1"](x_embed)
            x_embed = self.bns["blk_2_1"](x_embed)
            x_embed = self.relu(x_embed)

            x_embed = self.convs["blk_2_2"](x_embed)
            x_embed = self.bns["blk_2_2"](x_embed)
            x_embed = self.pool(x_embed)
            x_embed = self.relu(x_embed)

            x_embed = self.convs["blk_2_3"](x_embed)
            x_embed = self.bns["blk_2_3"](x_embed)
            x_embed = self.pool(x_embed)
            x_embed = self.relu(x_embed)

        # Forward propagate in FC layers of module_2
        x_embed = torch.reshape(x_embed, (x_embed.size(0), -1))
        x_embed = self.fc2(x_embed)

        if self.task == "binary":
            x_embed = self.bi_unit(x_embed)
        elif self.task == "multi":
            x_embed = self.mul_unit(x_embed)

        return x_embed
