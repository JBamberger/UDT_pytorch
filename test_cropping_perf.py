import numpy as np
import torch
import kornia
import cv2
import timeit


def crop_chw(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz - 1) / (bbox[2] - bbox[0])
    b = (out_sz - 1) / (bbox[3] - bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    out = np.transpose(crop, (2, 0, 1))
    out = torch.tensor(out).unsqueeze(0)
    return out


def crop_chw_torch(image, bbox, out_sz, device='cpu'):
    a = (out_sz - 1) / (bbox[2] - bbox[0])
    b = (out_sz - 1) / (bbox[3] - bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = torch.tensor([[a, 0, c], [0, b, d]], device=device).unsqueeze(0)

    return kornia.warp_affine(image, mapping, dsize=(out_sz, out_sz), )


if __name__ == '__main__':
    bbox = np.array([250, 250, 300, 300])
    out_size = 125

    x = np.random.randn(600, 600, 3)
    y: torch.tensor = kornia.image_to_tensor(x, keepdim=False) # .to('cuda')  # BxCxHxW


    # a = crop_chw(x, bbox, out_size)
    # b = crop_chw_torch(x, bbox, out_size)
    # print(a.shape)
    # print(b.shape)

    import torch.utils.benchmark as benchmark

    t0 = benchmark.Timer(
        stmt='crop_chw(x, box, 125)',
        setup='from __main__ import crop_chw',
        globals={'x': x, 'box':np.array([250, 250, 300, 300])})

    t1 = benchmark.Timer(
        stmt='crop_chw_torch(x, box, 125, device="cpu")',
        setup='from __main__ import crop_chw_torch',
        globals={'x': y, 'box': np.array([250, 250, 300, 300])})

    print(t0.timeit(1000))
    print(t1.timeit(1000))



    # assert a.allclose(b)

    pass
