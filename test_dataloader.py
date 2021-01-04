from dataset import ILSVRC2015
import torch


def main():
    ds = ILSVRC2015(train=True, range=10)
    # print(ds.mean.shape)
    # a,b,c = ds[1]
    # print(a.shape)
    # exit()

    dl = torch.utils.data.DataLoader(
        ds, batch_size=32, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    for i, (template, search1, search2) in enumerate(dl):
        print(f'{i}: Template: {template.shape}, Search1: {search1.shape}, Search2: {search2.shape}')


if __name__ == '__main__':
    main()
