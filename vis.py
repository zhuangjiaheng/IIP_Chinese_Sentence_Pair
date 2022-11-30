import matplotlib.pyplot as plt
plt.style.use("ggplot")
import numpy as np
from utils import save_json_file, load_json_file


aug_result_0 = load_json_file("./data/experiments/aug/fold0_results.json")
aug_result_1 = load_json_file("./data/experiments/aug/fold1_results.json")
aug_result_2 = load_json_file("./data/experiments/aug/fold2_results.json")
base_result = load_json_file("./data/experiments/baseline/results.json")
pgd_result = load_json_file("data/experiments/pgd/results.json")
fgm_result = load_json_file("./data/experiments/fgm/results.json")
aug_pgd_result_0 = load_json_file("./data/experiments/aug_pgd/fold0_results.json")
aug_pgd_result_1 = load_json_file("./data/experiments/aug_pgd/fold1_results.json")
aug_pgd_result_2 = load_json_file("./data/experiments/aug_pgd/fold2_results.json")


x = np.arange(1, 6)
aug0_train_acc, aug1_train_acc, aug2_train_acc = [], [], []
aug0_val_acc, aug1_val_acc, aug2_val_acc = [], [], []

base_train_acc, pgd_train_acc, fgm_train_acc = [], [], []
base_val_acc, pgd_val_acc, fgm_val_acc = [], [], []

aug0_pgd_train_acc, aug1_pgd_train_acc, aug2_pgd_train_acc = [], [], []
aug0_pgd_val_acc, aug1_pgd_val_acc, aug2_pgd_val_acc = [], [], []

for i in range(1, 6):
    aug0_train_acc.append(aug_result_0[str(i)][1])
    aug0_val_acc.append(aug_result_0[str(i)][3])
    aug1_train_acc.append(aug_result_1[str(i)][1])
    aug1_val_acc.append(aug_result_1[str(i)][3])
    aug2_train_acc.append(aug_result_2[str(i)][1])
    aug2_val_acc.append(aug_result_2[str(i)][3])

    base_train_acc.append(base_result[str(i)][1])
    base_val_acc.append(base_result[str(i)][3])
    pgd_train_acc.append(pgd_result[str(i)][1])
    pgd_val_acc.append(pgd_result[str(i)][3])
    fgm_train_acc.append(fgm_result[str(i)][1])
    fgm_val_acc.append(fgm_result[str(i)][3])

    aug0_pgd_train_acc.append(aug_pgd_result_0[str(i)][1])
    aug0_pgd_val_acc.append(aug_pgd_result_0[str(i)][3])
    aug1_pgd_train_acc.append(aug_pgd_result_1[str(i)][1])
    aug1_pgd_val_acc.append(aug_pgd_result_1[str(i)][3])
    aug2_pgd_train_acc.append(aug_pgd_result_2[str(i)][1])
    aug2_pgd_val_acc.append(aug_pgd_result_2[str(i)][3])


aug_train_acc = [(aug0_train_acc[i] + aug1_train_acc[i] + aug2_train_acc[i])/3 for i in range(5)]
aug_val_acc = [(aug0_val_acc[i] + aug1_val_acc[i] + aug2_val_acc[i])/3 for i in range(5)]
aug_pgd_train_acc = [(aug0_pgd_train_acc[i] + aug1_pgd_train_acc[i] + aug2_pgd_train_acc[i])/3 for i in range(5)]
aug_pgd_val_acc = [(aug0_pgd_val_acc[i] + aug1_pgd_val_acc[i] + aug2_pgd_val_acc[i])/3 for i in range(5)]

# print("baseline", max(base_train_acc), max(base_val_acc))
# print("fgm", max(fgm_train_acc), max(fgm_val_acc))
# print("pgd", max(pgd_train_acc), max(pgd_val_acc))
# print("augment", max(aug_train_acc), max(aug_val_acc))
# print("aug_pgd", max(aug_pgd_train_acc), max(aug_pgd_val_acc))
print(fgm_train_acc)


def plot_cv():
    """交叉验证"""
    plt.plot(x, aug0_train_acc, "b--", label="fold0/train")
    plt.plot(x, aug0_val_acc, "b.-", label="fold0/val")
    plt.plot(x, aug1_train_acc, "r--", label="fold1/train")
    plt.plot(x, aug1_val_acc, "r.-", label="fold1/val")
    plt.plot(x, aug2_train_acc, "g--", label="fold2/train")
    plt.plot(x, aug2_val_acc, "g.-", label="fold2/val")
    plt.legend()
    plt.xticks([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    plt.title("Cross Valid Result with Data Augmentation")
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.show()


def plot_adversarial():
    plt.plot(x, base_train_acc, "b--", label="baseline/train")
    plt.plot(x, base_val_acc, "b.-", label="baseline/val")
    plt.plot(x, fgm_train_acc, "r--", label="fgm/train")
    plt.plot(x, fgm_val_acc, "r.-", label="fgm/val")
    plt.plot(x, pgd_train_acc[:5], "g--", label="pgd/train")
    plt.plot(x, pgd_val_acc[:5], "g.-", label="pgd/val")
    plt.legend()
    plt.xticks([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    plt.title("Result with Adversarial Training")
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.show()


def plot_final():
    plt.plot(x, base_train_acc, "b--", label="baseline/train")
    plt.plot(x, base_val_acc, "b.-", label="baseline/val")
    plt.plot(x, pgd_train_acc[:5], "r--", label="pgd/train")
    plt.plot(x, pgd_val_acc[:5], "r.-", label="pgd/val")
    plt.plot(x, aug_train_acc, "g--", label="aug/train")
    plt.plot(x, aug_val_acc, "g.-", label="aug/val")
    plt.plot(x, aug_pgd_train_acc, "m--", label="aug+pgd/train")
    plt.plot(x, aug_pgd_val_acc, "m.-", label="aug+pgd/val")
    plt.legend()
    plt.xticks([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    plt.title("Results")
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.show()


if __name__ == '__main__':
    # plot_adversarial()
    plot_final()