import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
plt.style.use("ggplot")


class PreAnalysis:
    def __init__(self, train_path, dev_path):
        self.train_path = train_path
        self.dev_path = dev_path
        self.train, self.dev = self.load_csv_data()

    def load_csv_data(self):
        df_train = pd.read_csv(self.train_path)
        df_dev = pd.read_csv(self.dev_path)
        return df_train, df_dev

    def count_seq_len(self, mode="train"):
        if mode == "train":
            df = self.train
        else:
            df = self.dev
        q1_lengths, q2_lengths = [], []
        for ind, row in df.iterrows():
            s1, s2 = row["query1"], row["query2"]
            l1, l2 = len(s1), len(s2)
            q1_lengths.append(l1)
            q2_lengths.append(l2)
        return {"q1": q1_lengths, "q2": q2_lengths}

    def plot_seq_len(self):
        """query长度"""
        save_file = "figures/seq_len"
        train_len = self.count_seq_len(mode="train")
        dev_len = self.count_seq_len(mode="dev")
        train_q1_len, train_q2_len = train_len["q1"], train_len["q2"]
        dev_q1_len, dev_q2_len = dev_len["q1"], dev_len["q2"]
        sns.distplot(train_q1_len, hist=False, kde=False, fit=stats.norm,
                     fit_kws={'color': 'black', 'label': 'train_query1', 'linestyle': '-'})
        sns.distplot(train_q2_len, hist=False, kde=False, fit=stats.norm,
                     fit_kws={'color': 'blue', 'label': 'train_query2', 'linestyle': '-'})
        sns.distplot(dev_q1_len, hist=False, kde=False, fit=stats.norm,
                     fit_kws={'color': 'red', 'label': 'dev_query1', 'linestyle': '-'})
        sns.distplot(dev_q2_len, hist=False, kde=False, fit=stats.norm,
                     fit_kws={'color': 'orange', 'label': 'dev_query2', 'linestyle': '-'})
        plt.legend()
        plt.title("Length of Query")
        plt.savefig(save_file)
        plt.show()

    def count_category(self, mode="train") -> dict:
        if mode == "train":
            group = self.train.groupby("label").count()
        else:
            group = self.dev.groupby("label").count()
        res = {0: group["id"][0], 1: group["id"][1]}  # print(g["id"][0])
        return res

    def plot_category(self, mode="train"):
        save_file = "figures/train_category.jpg" if mode == "train" else \
            "figures/dev_category.jpg"
        cnt_res = self.count_category(mode)
        plt.bar(cnt_res.keys(), cnt_res.values(), width=0.2)
        plt.xticks([0, 1], [0, 1])
        plt.title(f"{mode} category")
        plt.savefig(save_file)
        plt.show()

    def similarity(self, mode="train"):
        if mode == "train":
            train_len = self.count_seq_len(mode="train")
            train_q1_len, train_q2_len = train_len["q1"], train_len["q2"]
            df = pd.DataFrame(np.array([train_q1_len, train_q2_len]).T,
                              columns=["q1", "q2"])
        else:
            dev_len = self.count_seq_len(mode="dev")
            dev_q1_len, dev_q2_len = dev_len["q1"], dev_len["q2"]
            df = pd.DataFrame(np.array([dev_q1_len, dev_q2_len]).T,
                              columns=["q1", "q2"])
        with sns.axes_style("dark"):
            sns.jointplot(x="q1", y="q2", data=df, kind="hex")
        plt.xlabel("length of query 1")
        plt.ylabel("length of query 2")
        plt.savefig(f"figures/similarity_{mode}.jpg")
        plt.show()


if __name__ == '__main__':
    # train = pd.read_csv("./data/train.csv")
    # print(train.head())
    pa = PreAnalysis(train_path="./data/train.csv",
                     dev_path="./data/dev.csv")
    # pa.plot_category(mode="dev")
    # pa.plot_seq_len()
    # pa.similarity(mode="test")


