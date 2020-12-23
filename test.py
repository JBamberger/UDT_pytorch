from dataset import ILSVRC2015
import torch


def main():
    ds = ILSVRC2015(train=True, range=10)
    print(ds.mean.shape)
    a,b,c = ds[1]
    print(a.shape)
    exit()

    dl = torch.utils.data.DataLoader(
        ds, batch_size=32, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    i = 0
    for a,b,c in dl:



        print(f'{i}: {a.shape}')


if __name__ == '__main__':
    main()
