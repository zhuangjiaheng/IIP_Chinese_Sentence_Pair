import torch

SEQ_LEN = 40
batch_size = 16

device = torch.device("mps")
learning_rate = 1e-5
epochs = 5

output_file = 'data/experiments/aug_pgd/'

# 对抗学习
adversarial_method = "pgd"
K = 3  # if pgd

# 交叉验证
seed = 42
