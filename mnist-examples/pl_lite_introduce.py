"""
exampleが少ないのと他のとの差別化がよくわからないなあ。

https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.lite.LightningLite.html#pytorch_lightning.lite.LightningLite

1. LightningLiteを継承したクラスを作り、runメソッドをオーバーライドする
2. vanilla_pytorchで定義していたrunをrunメソッドとする
3. deviceの設定はモジュールで実行されるので削除する
"""

import argparse
import torch
import torch.optim as optim
import torch.nn as nn
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
    def run(self, hprasm):
        self.hprams = hprasm

        seed_everything(hprasm.seed)

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,))
            ])

        # 一つのprocessのみで実行されるようにするための処理
        if self.is_global_zero:
            MNIST("./shared", download=True)
        self.barrier()

        train_dataset = MNIST(root="./shared", train=True, transform=transform)
        test_dataset = MNIST(root="./shared", train=False, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=hparams.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=hparams.batch_size)

        # こうするとdistributed trainingに対応できる
        train_loader, test_loader = self.setup_dataloaders(train_loader, test_loader)

        # 明示的にcudaに送る必要はない
        model = Net()

        optimizer = optim.Adadelta(model.parameters(), lr=hparams.lr)

        # こうするとdistributed trainingに対応でき、modelが適切なdeviceに配置される
        model, optimizer = self.setup(model, optimizer)

        scheduler = StepLR(optimizer, step_size=1, gamma=hparams.gamma)

        # 手動でaccuracyを計算するのではなくtorchmetricesを使用する
        test_acc = Accuracy().to(self.device)

        for epoch in range(1, hparams.epochs + 1):
            model.train()

            for batch_idx, (data, target) in enumerate(train_loader):
                # 明示的にdeviceを設定する必要はない
                #data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                
                # クラスにあるbackwordメソッドを使用する
                #loss.backward()
                self.backward(loss)

                optimizer.step()

                if (batch_idx == 0) or ((batch_idx + 1) % hparams.log_interval == 0):

                    print(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch,
                            batch_idx * len(data),
                            len(train_loader.dataset),
                            100.0 * batch_idx / len(train_loader),
                            loss.item(),
                        )
                    )

                    if hparams.dry_run:
                        break
            
            scheduler.step()

            # TESTTING LOOP
            model.eval()
            test_loss = 0.
            with torch.no_grad():
                for data, target in test_loader:
                    output = model(data)
                    test_loss += F.nll_loss(output, target, reduction="sum").item()

                    # 自分でargmaxを取って計算をする必要がなくなる
                    test_acc(output, target)

                    if hparams.dry_run:
                        break
            
            # all_gatherでprocess毎のlossをすべて集計できる
            test_loss = self.all_gather(test_loss).sum() / len(test_loader.dataset)

            print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: ({test_acc.compute():.2f}%)\n")
            test_acc.reset()

            if hparams.dry_run:
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightningLite MNIST Example")
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument("--epochs", type=int, default=1, metavar="N", help="number of epochs to train (default: 14)")
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

    Lite(accelerator='auto', devices='auto').run(hparams)