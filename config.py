import os

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_root = os.path.join('D:', os.sep, 'ma', 'datasets')
checkpoint_root = os.path.join('D:', os.sep, 'ma', 'checkpoints')
results_root = os.path.join('D:', os.sep, 'ma', 'results')
