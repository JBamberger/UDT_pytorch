import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.autograd.profiler as profiler

import util

if __name__ == '__main__':
    n = 5
    s = 121
    y = torch.as_tensor(util.gaussian_shaped_labels(4, (s, s)))

    response = torch.zeros((n, 1, s, s))
    response[0, 0, 60, 60] = 100
    response[1, 0, 0, 60] = 100
    response[2, 0, 60, 0] = 100
    response[3, 0, 30, 100] = 100
    response[4, 0, 80, 90] = 100

    with profiler.profile(use_cuda=True, record_shapes=True, profile_memory=True, with_stack=True) as p:
        # with profiler.record_function('model_inference'):
        fake_y = util.create_fake_y(y, response)

    print(p.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=-1))

    fig, ax = plt.subplots(2, n)
    for i in range(n):
        ax[0, i].imshow(response[i, 0, ...].numpy())
        ax[1, i].imshow(fake_y[i, 0, ...].numpy())

    fig.show()
