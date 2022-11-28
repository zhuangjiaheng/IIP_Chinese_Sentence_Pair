import torch

SEQ_LEN = 40
batch_size = 16

device = torch.device("mps")
learning_rate = 1e-5
epochs = 5

output_file = './data/experiments/baseline/'
