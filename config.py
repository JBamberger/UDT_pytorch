import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_root = "D:/ma/datasets/"
checkpoint_root = "D:/ma/checkpoints/"
