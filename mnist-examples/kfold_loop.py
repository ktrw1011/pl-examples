import os
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type
from numpy import void

import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, Subset

import torchvision.transforms as T
from torchvision.datasets import MNIST

from sklearn.model_selection import KFold

from pytorch_lightning import LightningDataModule, seed_everything, Trainer, LightningModule
from pytorch_lightning.loops.base import Loop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.trainer.states import TrainerFn

from pl_lightningmodule import ImageClassifier


seed_everything(42)

# KFoldDataModuleの抽象クラスを定義
class BaseKFoldDataModule(LightningDataModule, ABC):
    @abstractmethod
    def setup_folds(self, num_folds:int): ...
    
    @abstractmethod
    def setup_fold_index(self, fold_index:int): ...


# KFoldDataModuleの抽象クラスを継承したクラスを定義
class MNISTKFoldDataModule(BaseKFoldDataModule):

    train_dataset: Optional[Dataset] = None
    test_dataset: Optional[Dataset] = None
    train_fold: Optional[Dataset] = None
    val_fold: Optional[Dataset] = None

    def prepare_data(self) -> None:
        # データのダウンロード
        MNIST("./shared", transform=T.Compose([T.ToTensor(), T.Normalize(mean=(0.5,), std=(0.5,))]))

    def setup(self, stage: Optional[str] = None) -> None:
        # load the data
        dataset = MNIST("./shared", transform=T.Compose([T.ToTensor(), T.Normalize(mean=(0.5,), std=(0.5,))]))
        
        # testを用意
        self.train_dataset, self.test_dataset = random_split(dataset, [50000, 10000])

    def setup_folds(self, num_folds: int):
        self.num_folds = num_folds
        self.splits = list(KFold(num_folds).split(range(len(self.train_dataset))))

    def setup_fold_index(self, fold_index: int):
        train_indices, val_indices = self.splits[fold_index]
        self.train_fold = Subset(self.train_dataset, train_indices)
        self.val_fold = Subset(self.train_dataset, val_indices)

    def train_dataloader(self):
        return DataLoader(self.train_fold, batch_size=128)

    def val_dataloader(self):
        return DataLoader(self.val_fold, batch_size=128)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=128)
    
# foldのcheckpointからmodelを呼び出して、推論を行う
# これはおそらくoofではなくtestに対する推論
class EmsambleVotingModel(LightningModule):
    def __init__(self, model_cls: Type[LightningModule], checkpoint_paths:List[str]):
        super().__init__()
        self.models = torch.nn.ModuleList([model_cls.load_from_checkpoint(p) for p in checkpoint_paths])

    def test_step(self, batch:Any, batch_idx: int, dataloader_idx: int=0) -> None:
        logits = torch.stack([m(batch[0]) for m in self.models]).mean(dim=0)
        loss = F.cross_entropy(logits, batch[1])
        self.log("test_loss", loss)

# v.15からloopクラスが追加された
#############################################################################################
#                     Here is the `Pseudo Code` for the base Loop.                          #
# class Loop:                                                                               #
#                                                                                           #
#   def run(self, ...):                                                                     #
#       self.reset(...)                                                                     #
#       self.on_run_start(...)                                                              #
#                                                                                           #
#        while not self.done:                                                               #
#            self.on_advance_start(...)                                                     #
#            self.advance(...)                                                              #
#            self.on_advance_end(...)                                                       #
#                                                                                           #
#        return self.on_run_end(...)                                                        #
#############################################################################################
class KFoldLoop(Loop):

    def __init__(self, num_folds: int, fit_loop:FitLoop, export_path: str) -> None:
        super().__init__()

        self.num_folds = num_folds
        self.fit_loop = fit_loop
        self.current_fold: int = 0
        self.export_path = export_path

    # これがFalseになるまでLoopを続ける
    @property
    def done(self) -> bool:
        return self.current_fold >= self.num_folds

    def reset(self) -> None:
        """なにもしない"""

    def _reset_fitting(self) -> None:
        # ここでdataloaderリセットしているけどこれは？
        self.trainer.reset_train_dataloader()
        self.trainer.reset_val_dataloader()
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True

    def _reset_testing(self) -> None:
        self.trainer.reset_test_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True

    def on_run_start(self, *args:Any, **kwargs:Any) -> None:
        """`BaseKFoldDataModule`から`setup_folds`を呼ぶ"""
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        
        # ここでFoldを分割している
        self.trainer.datamodule.setup_folds(self.num_folds)
        
        # これはお作法？
        # self.lightning_module_state_dict = deepcopy(self.trainer.lightning_module.state_dict())

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        print(f"STARTING FOLD {self.current_fold}")
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_fold_index(self.current_fold)

    def advance(self, *args: Any, **kwargs: Any) -> None:
        self._reset_fitting()
        
        # fitのloopを呼ぶ
        self.fit_loop.run()

        self._reset_testing()
        
        # test_stepを呼ぶ
        self.trainer.test_loop.run()
        self.current_fold += 1

    def on_advance_end(self) -> None:
        self.trainer.save_checkpoint(os.path.join(self.export_path, f"model.{self.current_fold}.pt"))

        # weightとoptimizerをリストアする
        # validationのため？
        # self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)
        # self.trainer.accelerator.setup_optimizers(self.trainer)
    
    def on_run_end(self) -> None:
        # checkpointを呼び出して、アンサンブルのモデルでtestを推論している
        # かなりお作法が多い。ここは別途スクラッチしてもいいのではという感じ
        checkpoint_paths = [os.path.join(self.export_path, f"model.{f_idx+1}.pt") for f_idx in range(self.num_folds)]
        
        voting_model = EmsambleVotingModel(type(self.trainer.lightning_module), checkpoint_paths)
        voting_model.trainer = self.trainer

        self.trainer.accelerator.connect(voting_model)
        self.trainer.training_type_plugin.model_to_device()
        self.trainer.test_loop.run()

    def on_save_checkpoint(self) -> Dict[str, int]:
        return {"current_fold": self.current_fold}

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        self.current_fold = state_dict["current_fold"]

    def __getattr__(self, key) -> Any:
        # requires to be overridden as attributes of the wrapped loop are being accessed.
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]

if __name__ == "__main__":
    model = ImageClassifier()
    datamodule = MNISTKFoldDataModule()

    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu",
        num_sanity_val_steps=0,
    )

    trainer.fit_loop = KFoldLoop(num_folds=5, fit_loop=trainer.fit_loop, export_path="./")
    trainer.fit(model, datamodule)