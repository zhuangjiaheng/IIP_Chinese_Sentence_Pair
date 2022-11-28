import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from DataProcesser import train_loader, dev_loader
from Model import BertClassifier
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
import transformers
from Config import *
from utils import load_json_file, save_json_file

transformers.logging.set_verbosity_error()


if __name__ == '__main__':

    model = BertClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(params=model.parameters(), lr=learning_rate)

    best_val_acc, best_epoch = -1, -1
    results = dict()
    for epoch in range(1, epochs+1):
        train_acc, train_loss, train_len = 0, 0, 0
        dev_acc, dev_loss, dev_len = 0, 0, 0
        dev_predict, dev_gt = [], []
        model.train()
        ttq = tqdm(train_loader)
        for data in ttq:
            input_ids = data["input_ids"].to(device)
            input_masks = data["attention_mask"].to(device)
            segment_ids = data["token_type_ids"].to(device)
            label = data["targets"].to(device)

            y_pred = model(input_ids, input_masks, segment_ids)
            loss = criterion(y_pred, label)
            loss.backward()
            optimizer.step()
            model.zero_grad()

            y_max = np.argmax(y_pred.detach().to("cpu").numpy(), axis=1)
            label = label.detach().to("cpu").numpy()

            train_acc += sum(y_max == label)
            train_loss += loss.item()
            train_len += len(label)
            ttq.set_postfix(epoch=epoch, loss=train_loss / train_len, acc=train_acc / train_len)

        model.eval()
        dtq = tqdm(dev_loader)
        for data in tqdm(dev_loader):
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
            dtq.set_postfix(epoch=epoch, loss=dev_loss / dev_len, acc=dev_acc / dev_len)

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
            torch.save(model.state_dict(), output_file + 'best_model_state.pt')

    save_json_file(results, output_file + "results.json")
