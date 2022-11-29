import pandas as pd
from Config import *
import torch
from torch import nn
from DataProcesser import Covid19Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers import BertTokenizer, BertConfig, AdamW
from Adversarial import PGD, FGM
from Model import BertClassifier
from sklearn.metrics import classification_report
from tqdm import tqdm
import numpy as np
from utils import save_json_file
import transformers
transformers.logging.set_verbosity_error()


model_path = './data/pretrained_model/chinese_roberta_wwm_large_ext_pytorch/'
bert_config = BertConfig.from_pretrained(model_path + 'bert_config.json', output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained(model_path + 'vocab.txt', config=bert_config)

model = BertClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(params=model.parameters(), lr=learning_rate)
pgd = PGD(model)
fgm = FGM(model)

# for fold, (train_index, valid_index) in enumerate(skf.split(categories, labels)):
#     print('\n\n------------fold:{}------------\n'.format(fold))
#     c = categories[train_index]
#     q1 = query1[train_index]
#     q2 = query2[train_index]
#     y = labels[train_index]
#
#     val_c = categories[valid_index]
#     val_q1 = query1[valid_index]
#     val_q2 = query2[valid_index]
#     val_y = labels[valid_index]
#
#     train_td = Covid19Dataset(q1, q2, c, y, tokenizer, SEQ_LEN)
#     dev_td = Covid19Dataset(val_q1, val_q2, val_c, val_y, tokenizer, SEQ_LEN)
#     aug_train_loader = DataLoader(train_td, batch_size=batch_size, num_workers=4)
#     aug_dev_loader = DataLoader(dev_td, batch_size=batch_size, num_workers=4)


if __name__ == '__main__':
    train_dev_data = pd.read_csv("./data/datasets/train_dev_augment.csv")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    query1 = train_dev_data["query1"].values
    query2 = train_dev_data["query2"].values
    categories = train_dev_data["category"].values
    labels = train_dev_data["label"].values

    for fold, (train_index, valid_index) in enumerate(skf.split(categories, labels)):
        print('\n------------fold:{}------------\n'.format(fold))
        c = categories[train_index]
        q1 = query1[train_index]
        q2 = query2[train_index]
        y = labels[train_index]

        val_c = categories[valid_index]
        val_q1 = query1[valid_index]
        val_q2 = query2[valid_index]
        val_y = labels[valid_index]

        train_td = Covid19Dataset(q1, q2, c, y, tokenizer, SEQ_LEN)
        dev_td = Covid19Dataset(val_q1, val_q2, val_c, val_y, tokenizer, SEQ_LEN)
        aug_train_loader = DataLoader(train_td, batch_size=batch_size, num_workers=4)
        aug_dev_loader = DataLoader(dev_td, batch_size=batch_size, num_workers=4)

        best_val_acc, best_epoch = -1, -1
        results = dict()
        for epoch in range(1, epochs + 1):
            train_acc, train_loss, train_len = 0, 0, 0
            dev_acc, dev_loss, dev_len = 0, 0, 0
            dev_predict, dev_gt = [], []
            model.train()
            ttq = tqdm(aug_train_loader)
            for data in ttq:
                input_ids = data["input_ids"].to(device)
                input_masks = data["attention_mask"].to(device)
                segment_ids = data["token_type_ids"].to(device)
                label = data["targets"].to(device)

                y_pred = model(input_ids, input_masks, segment_ids)
                loss = criterion(y_pred, label)
                loss.backward()

                # 对抗训练
                if adversarial_method == "pgd":
                    pgd.backup_grad()
                    for t in range(K):
                        pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                        if t != K - 1:
                            model.zero_grad()
                        else:
                            pgd.restore_grad()
                        y_pred = model(input_ids, input_masks, segment_ids)

                        loss_adv = criterion(y_pred, label)
                        loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    pgd.restore()  # 恢复embedding参数
                elif adversarial_method == "fgm":
                    fgm.attack()  # 在embedding上添加对抗扰动
                    y_pred = model(input_ids, input_masks, segment_ids)
                    loss_adv = criterion(y_pred, label)
                    loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    fgm.restore()  # 恢复embedding参数

                optimizer.step()
                model.zero_grad()

                y_max = np.argmax(y_pred.detach().to("cpu").numpy(), axis=1)
                label = label.detach().to("cpu").numpy()

                train_acc += sum(y_max == label)
                train_loss += loss.item()
                train_len += len(label)
                ttq.set_postfix(epoch=epoch, loss=train_loss / train_len, acc=train_acc / train_len)

            model.eval()
            dtq = tqdm(aug_dev_loader)
            for data in dtq:
                input_ids = data["input_ids"].to(device)
                input_masks = data["attention_mask"].to(device)
                segment_ids = data["token_type_ids"].to(device)
                label = data["targets"].to(device)
                y_pred = model(input_ids, input_masks, segment_ids)
                loss = criterion(y_pred, label)

                y_max = np.argmax(y_pred.detach().to("cpu").numpy(), axis=1)
                label = label.detach().to("cpu").numpy()
                dev_acc += sum(y_max == label)

                dev_gt.extend(label)
                dev_predict.extend(y_max)

                dev_loss += loss.item()
                dev_len += len(label)
                # dtq.set_postfix(epoch=epoch, loss=dev_loss / dev_len, acc=dev_acc / dev_len)

            # show result
            train_loss = train_loss / train_len
            train_acc = train_acc / train_len
            dev_loss = dev_loss / dev_len
            dev_acc = dev_acc / dev_len
            results[epoch] = [train_loss, train_acc, dev_loss, dev_acc]
            print(f"- {epoch}/{epochs}")
            print(f"Train Loss:{train_loss}, Train Acc:{train_acc}, Val Loss:{dev_loss}, Val Acc:{dev_acc}")
            print(classification_report(dev_predict, dev_gt))

            if dev_acc > best_val_acc:
                best_val_acc = dev_acc
                best_epoch = epoch
                torch.save(model.state_dict(), output_file + f'fold{fold}_best_model_state.pt')

        save_json_file(results, output_file + f"fold{fold}_results.json")
