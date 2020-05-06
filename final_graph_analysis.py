import pandas as pd
import matplotlib.pyplot as plt

train_dataset = ["m30k_deen_shr_256.train.log", "m30k_deen_shr_256_1024.train.log", "m30k_deen_shr_256_4096.train.log", "m30k_deen_shr_512_4096.train.log", "m30k_deen_shr_1024.train.log", "m30k_deen_shr.mlp.train.log","m30k_deen_shr.baseline.train.log"]
train_labels = ["d_model=256, d_hidden=2048", "d_model=256, d_hidden=1024", "d_model=256, d_hidden=4096", "d_model=512, d_hidden=4096", "d_model=512, d_hidden=1024", "d_model=512, d_hidden=2048", "baseline"]
for i in range(len(train_dataset)):
    train_data = pd.read_csv(train_dataset[i], sep=",")
    train_epoch = train_data['epoch'].values
    train_acc = train_data['accuracy'].values
    plt.plot(train_epoch, train_acc, label=train_labels[i])
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy VS Epoch on Training Data')

plt.show()

valid_dataset = ["m30k_deen_shr_256.valid.log", "m30k_deen_shr_256_1024.valid.log", "m30k_deen_shr_256_4096.valid.log", "m30k_deen_shr_512_4096.valid.log", "m30k_deen_shr_1024.valid.log", "m30k_deen_shr.mlp.valid.log","m30k_deen_shr.baseline.valid.log"]
valid_labels = ["d_model=256, d_hidden=2048", "d_model=256, d_hidden=1024", "d_model=256, d_hidden=4096", "d_model=512, d_hidden=4096", "d_model=512, d_hidden=1024", "d_model=512, d_hidden=2048", "baseline"]
for i in range(len(valid_dataset)):
    valid_data = pd.read_csv(valid_dataset[i], sep=",")
    valid_epoch = valid_data['epoch'].values
    valid_acc = valid_data['accuracy'].values
    plt.plot(valid_epoch, valid_acc, label=valid_labels[i])
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy VS Epoch on Validation Data')

plt.show()