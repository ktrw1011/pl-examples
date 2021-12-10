"""
loopが隠蔽されたので全体的にスッキリする。loop内をスクラッチしたい場合は柔軟性にかける
バリデーションステップがないので他のexampleとは微妙にフローが異なっている
"""

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.optim.lr_scheduler import StepLR

from torchmetrics import Accuracy

import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.callbacks import RichProgressBar


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3 ,1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class ImageClassifier(pl.LightningModule):

    def __init__(self, model=None, lr=1.0, gamma=0.7, batch_size=32):
        super().__init__()

        self.save_hyperparameters()
        self.model = model or Net()
        self.test_acc = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        self.log("test_acc", self.test_acc(logits, y))
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adadelta(self.model.parameters(), lr=self.hparams.lr)
        return [optimizer], [StepLR(optimizer, step_size=1, gamma=self.hparams.gamma)]

    @property
    def transform(self):
         return T.Compose([T.ToTensor(),T.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self) -> None:
         MNIST("./shared", download=True)

    def train_dataloader(self):
        train_dataset = MNIST("./shared", train=True, download=False, transform=self.transform)
        return DataLoader(train_dataset, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        test_dataset = MNIST("./shared", train=False, download=False, transform=self.transform)
        return DataLoader(test_dataset, batch_size=self.hparams.batch_size)

def cli_main():
    
    cli = LightningCLI(
        ImageClassifier,
        seed_everything_default=42,
        save_config_overwrite=True,
        run=False,
    )

    cli.trainer.fit(
        cli.model, datamodule=cli.datamodule
    )

    cli.trainer.test(
        ckpt_path="best",
        datamodule=cli.datamodule
    )

if __name__ == "__main__":
    cli_main()