from typing import Tuple

import torch


def unravel_index(indices: torch.LongTensor, shape: Tuple[int, ...], ) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        unravel coordinates, (*, N, D).

    Source:
        https://github.com/pytorch/pytorch/issues/35674#issuecomment-739492875
    """

    shape = torch.tensor(shape)
    indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = torch.zeros(indices.size() + shape.size(), dtype=int)

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = indices // dim

    return coord.flip(-1)


if __name__ == '__main__':
    import numpy as np

    indices = np.array([5, 6, 7, 99])
    shape = (10, 10)

    npcoords = np.unravel_index(indices, shape)
    print(npcoords)
    print(np.stack(npcoords, axis=1))
    torchcoords = unravel_index(torch.tensor(indices), shape)
    print(torchcoords)
    print(torchcoords[...,0])
    print(torchcoords[...,1])
    print(tuple(torchcoords.t()))

    pass
