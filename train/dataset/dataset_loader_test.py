# Create training dataset loader
import torch
from tqdm import tqdm

from dataset import ILSVRC2015

train_dataset = ILSVRC2015(train=True, range=10)
val_dataset = ILSVRC2015(train=False, range=10)

if __name__ == '__main__':
    workers = 8

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True,
        num_workers=workers, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False,
        num_workers=workers, pin_memory=True, drop_last=True)

    for _ in tqdm(train_loader):
        pass

    for _ in tqdm(val_loader):
        pass
