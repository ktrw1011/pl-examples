"""
LightningLiteとLightningModuleの中間
基本的にはpl_light_introduce.pyと同じだが、trainingのstep等を関数化させている
trainからtestへのワークフローやデータ供給のloopは自前で書くので柔軟性は高い

https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.lite.LightningLite.html#pytorch_lightning.lite.LightningLite
"""

import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import optimizer
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.optim.lr_scheduler import StepLR

from torchmetrics import Accuracy
from pytorch_lightning import seed_everything
from pytorch_lightning.lite import LightningLite

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


class Lite(LightningLite):

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

    def run(self, hparams):
        self.hparams = hparams
        
        seed_everything(hparams.seed)

        self.model = Net()
        [optimizer], [scheduler] = self.configure_optimizers()
        model, optimizer = self.setup(self.model, optimizer)

        if self.is_global_zero:
            self.prepare_data()

        train_loader, test_loader = self.setup_dataloaders(self.train_dataloader(), self.test_dataloader())

        self.test_acc = Accuracy().to(self.device)

        for epoch in range(1, hparams.epochs + 1):
            self.model.train()
            
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                loss = self.training_step(batch, batch_idx)
                self.backward(loss)
                optimizer.step()

                if (batch_idx == 0) or ((batch_idx + 1) % hparams.log_interval == 0):

                    print(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch,
                            (batch_idx + 1) * self.hparams.batch_size,
                            len(train_loader.dataset),
                            100.0 * batch_idx / len(train_loader),
                            loss.item(),
                        )
                    )

                    if hparams.dry_run:
                        break
            
            scheduler.step()
            
            # TESTTING LOOP
            self.model.eval()
            test_loss = 0.
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    test_loss += self.test_step(batch, batch_idx)
                    if hparams.dry_run:
                        break
            
            test_loss = self.all_gather(test_loss).sum() / len(test_loader.dataset)
            print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: ({self.test_acc.compute():.2%}%)\n")

            self.test_acc.reset()
            if hparams.dry_run:
                break

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
        self.test_acc(logits, y)
        return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightningLite to LightningModule MNIST Example")
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument("--epochs", type=int, default=14, metavar="N", help="number of epochs to train (default: 14)")
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M", help="Learning rate step gamma (default: 0.7)")
    parser.add_argument("--dry-run", action="store_true", default=False, help="quickly check a single pass")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--save-model", action="store_true", default=False, help="For Saving the current Model")
    hparams = parser.parse_args()

    Lite(accelerator="auto", devices="auto").run(hparams)