# IIP_Chinese_Sentence_Pair
The Third Experiment of the Intelligent Information Processing Course



### 代码运行

- 若采用原始数据进行训练，确保`data/datasets`路径下含有`train.csv`, `dev.csv`，`test.csv`文件，并直接运行`train.py`
- 若采用增强后的数据进行训练，确保`data/datasets`路径下有`train_dev_augment.csv`文件，并直接运行`AugmentTrain.py`



### 代码架构

```
.
├── Adversarial.py            # 提供FGM和PGD两种对抗学习方法
├── AugmentTrain.py           # 训练脚本2:使用增强后的数据进行交叉验证
├── Config.py                 # 相关参数
├── DataProcesser.py          # 将数据组batch
├── EDA.py                    # 数据预处理和EDA分析
├── Model.py                  # Bert模型结构
├── README.md            
├── data                      
│   ├── datasets              # 存放需要读入的数据（原始数据和增强数据）
│   │   ├── dev.csv
│   │   ├── test.csv
│   │   ├── train.csv
│   │   └── train_dev_augment.csv  # 增强数据
│   ├── experiments           # 存放模型结果，包括最优模型和loss及准确率的记录
│   │   ├── aug               # 数据增强
│   │   ├── aug_pgd           # 数据增强+pgd对抗训练
│   │   ├── baseline          # baseline
│   │   │   ├── best_model_state.pt
│   │   │   └── results.json
│   │   ├── fgm               # fgm对抗训练
│   │   └── pgd               # pgd对抗训练
│   └── pretrained_model      # 存放预训练模型，采用Roberta_wwm_large
│       └── chinese_roberta_wwm_large_ext_pytorch
│           ├── bert_config.json
│           ├── pytorch_model.bin
│           └── vocab.txt
├── figures                   # 存放绘图结果
├── train.py                  # 训练脚本1:在原数据上进行巡来呢
├── utils.py                  # 提供json读取函数
├── vis.py                    # 结果可视化
```



### 模型结果

| 模型         | 验证集准确率 |
| ------------ | ------------ |
| baseline     | 0.9356       |
| baseline+fgm | 0.9365       |
| baseline+pgd | 0.9370       |
| 数据增强     | 0.9768       |
| 数据增强+pgd | **0.9797**   |

