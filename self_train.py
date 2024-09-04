import h5py
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model.vae_self_train import ModelVAE
import random
from matplotlib import pyplot as plt
class MyDataset(Dataset):
    def __init__(self, trainX, trainY, split_ratio):
        N = trainX.shape[0]

        TrainNum = int((N * (1 - split_ratio)))
        self.x = trainX[0:13000].astype(np.float32)
        self.y = trainY[0:13000].astype(np.float32)

        self.len = len(self.y)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        return (x, y)


class MyTestset(Dataset):
    def __init__(self, trainX, trainY, split_ratio):
        N = trainX.shape[0]

        TrainNum = int((N * (1 - split_ratio)))
        self.x = trainX[13000:13500].astype(np.float32)
        self.y = trainY[13000:13500].astype(np.float32)

        self.len = len(self.y)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        return (x, y)


BATCH_SIZE = 200
LEARNING_RATE = 0.001
TOTAL_EPOCHS = 500
split_ratio = 0.05
change_learning_rate_epochs = 10

model_save = 'pre_90.pth'

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")

if __name__ == '__main__':

    file_name1 = 'data/Case_1_2_Training.npy'
    # file_name1 = 'data/Case_1_2_Training.npy'
    print('The current dataset is : %s' % (file_name1))
    CIR = np.load(file_name1)
    trainX = CIR.transpose((2, 1, 3, 0))  # [none, 256, 72, 2]
    # trainX=trainX[0:1000]
    file_name2 = 'data/Case_1_2_Training_Label.npy'
    # file_name2 = 'data/Case_1_2_Training_Label.npy'
    print('The current dataset is : %s' % (file_name2))
    POS = np.load(file_name2)
    trainY = POS.transpose((1, 0))  # [none, 2]
    trainY=trainY[0:1000]
    trainY=trainX


    model = ModelVAE()
    # model_dict=model.state_dict()
    # pre_dict=torch.load("modelSubmit_2.pth")
    # pre_dict={k:v for k,v in pre_dict.items() if k in model_dict }
    # model_dict.update(pre_dict)
    # model.load_state_dict(model_dict)
    # model_dict=model.state_dict()
    # pre_dict=torch.load("modelSubmit_2.pth")
    # #pre_dict=pre.state_dict()
    #
    # pre_dict = {k: v for k, v in pre_dict.items() if k in model_dict}
    #
    # model_dict.update(pre_dict)
    # model.load_state_dict(torch.load("pre.pth"))


    model = model.to(DEVICE)
    print(model)

    train_dataset = MyDataset(trainX, trainY, split_ratio)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)  # shuffle 标识要打乱顺序
    test_dataset = MyTestset(trainX, trainY, split_ratio)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=True)  # shuffle 标识要打乱顺序
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    test_avg_min = 10000;
    for epoch in range(TOTAL_EPOCHS):
        model.train()
        optimizer.param_groups[0]['lr'] = LEARNING_RATE / np.sqrt(np.sqrt(epoch +200))

        # Learning rate decay
        if (epoch + 1) % change_learning_rate_epochs == 0:
            optimizer.param_groups[0]['lr'] /= 2
            print('lr:%.4e' % optimizer.param_groups[0]['lr'])

        # Training in this epoch
        loss_avg = 0
        for i, (x, y) in enumerate(train_loader):
            x = x.float().to(DEVICE)
            #patch=int (random.random()*16+6)
            # patch=int (random.uniform(11,16))
            patch=64
            # patch=int (256*72*0.8)
            mask=torch.hstack([torch.zeros(patch), torch.ones(72-patch)])

            for i in range(x.shape[0]):
                mask = np.hstack([np.zeros(patch), np.ones(72 - patch)])
                np.random.shuffle(mask)
                # mask=np.roll(mask,int(random.uniform(0,16)))
                # mask=mask.reshape(256,72)
                # mask=np.expand_dims(mask,axis=2)
                # mask=mask.repeat(2,axis=2)
                #idx = torch.randperm(mask.nelement())

                #mask = mask[idx]
                # np.random.shuffle(mask)
                # mask=mask.repeat([4])
                mask=mask==1
                # p = mask.cuda()
                # p = torch.unsqueeze(p, axis=0)
                # # p = p.repeat([256,1])
                # p = torch.unsqueeze(p, axis=2)
                # # p = p.repeat([1,1,2])
                # p = torch.unsqueeze(p, axis=0)
                # p = p.repeat([1, 256, 1, 2])
                x[i, :, ~mask, :] = 0

                # x[i, ~mask] = 0



            y = y.float().to(DEVICE)

            # 清零
            optimizer.zero_grad()
            output = model(x)
            # 计算损失函数
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            loss_avg += loss.item()

        loss_avg /= len(train_loader)

        # Testing in this epoch
        model.eval()
        test_avg = 0
        for i, (x, y) in enumerate(test_loader):
            x = x.float().to(DEVICE)
            # patch=int(256*72*0.8)
            patch=64
            #mask=torch.hstack([torch.zeros(patch), torch.ones(18-patch)])


            for i in range(x.shape[0]):
                mask = np.hstack([np.zeros(patch), np.ones(72 - patch)])
                # np.random.shuffle(mask)
                mask=np.roll(mask,int (random.uniform(0,16)))
                # mask=mask.reshape(256,72)
                # mask=np.expand_dims(mask,axis=2)
                # mask=mask.repeat(2,axis=2)
                #idx = torch.randperm(mask.nelement())

                #mask = mask[idx]
                # np.random.shuffle(mask)
                # mask=mask.repeat([4])
                mask=mask==1
                # p = mask.cuda()
                # p = torch.unsqueeze(p, axis=0)
                # # p = p.repeat([256,1])
                # p = torch.unsqueeze(p, axis=2)
                # # p = p.repeat([1,1,2])
                # p = torch.unsqueeze(p, axis=0)
                # p = p.repeat([1, 256, 1, 2])
                x[i, :, ~mask, :] = 0

                # x[i, ~mask] = 0

                # x[i, ~mask] = 0
            y = y.float().to(DEVICE)

            output = model(x)
            # 计算损失函数
            loss_test = criterion(output, y)
            test_avg += loss_test.item()

        test_avg /= len(test_loader)
        print('Epoch : %d/%d, Loss: %.6f, Test: %.6f, BestTest: %.6f' % (
        epoch + 1, TOTAL_EPOCHS, loss_avg, test_avg, test_avg_min))
        if test_avg < test_avg_min:
            print('Model saved!')
            test_avg_min = test_avg


            torch.save(model.state_dict(), model_save)

else:
    print("load torch model")
