
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model.fine_tune_model import ModelVAE
import random

class MyDataset(Dataset):
    def __init__(self, trainX, trainY, split_ratio):
        N = trainX.shape[0]

        TrainNum = int((N * (1 - split_ratio)))
        self.x = trainX[0:10000].astype(np.float32)
        self.y = trainY[0:10000].astype(np.float32)


        self.len = len(self.y)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = self.x[idx]
        # if random.random()>0.4:
            # SNR = np.random.random() * 5
            # noise = np.random.randn(x.shape[0], x.shape[1], x.shape[2])  # 产生N(0,1)噪声数据
            # noise = noise - np.mean(noise)  # 均值为0
            # signal_power = np.linalg.norm(x - x.mean()) ** 2 / x.size  # 此处是信号的std**2
            # noise_variance = signal_power / np.power(10, (SNR / 10))  # 此处是噪声的std**2
            # noise = (np.sqrt(noise_variance) / np.std(noise)) * noise  ##此处是噪声的std**2
            # x = noise + x

            # SNR = 0
            # noise = np.random.randn(x.shape[0], x.shape[1], x.shape[2])  # 产生N(0,1)噪声数据
            # signal_power = np.linalg.norm(x) ** 2 / x.size  # 此处是信号的std**2
            # noise_variance = signal_power / np.power(10, (SNR / 10))  # 此处是噪声的std**2
            # noise = np.sqrt(noise_variance) * noise  ##此处是噪声的std**2
            # x = noise + x

        # if random.random()<0.5:
        #     x=data_enhance(x)
        # ra=random.random()
        # rb=(1-ra*ra)**0.5
        # xa=x[...,0]*ra-x[...,1]*rb
        # xb=x[...,0]*rb+x[...,1]*ra
        # x[...,0]=xa
        # x[...,1]=xb
        y = self.y[idx]

        return (x, y)


class MyTestset(Dataset):
    def __init__(self, trainX, trainY, split_ratio):
        N = trainX.shape[0]

        TrainNum = int((N * (1 - split_ratio)))
        self.x = trainX[10000:13500].astype(np.float32)
        self.y = trainY[10000:13500].astype(np.float32)

        self.len = len(self.y)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = self.x[idx]
        # 给数据加指定SNR的高斯噪声




        y = self.y[idx]

        return (x, y)
def data_enhance(x_in):
    x_out = x_in.copy()
    x_sig = np.ndarray(x_out.shape[:-1]).astype(np.complex64)
    x_sig.real = x_out[..., 0]
    x_sig.imag = x_out[..., 1]
    x_fft = torch.fft.fft(torch.tensor(x_sig).cuda(), dim=-2)
    noise = torch.normal(mean=torch.zeros(x_fft.shape),std=10e-4).to(x_fft.device)
    x_fft = x_fft + noise
    x_ifft = torch.fft.ifft(x_fft,dim=-2).cpu()
    x_out[..., 0] = x_ifft.real
    x_out[..., 1] = x_ifft.imag
    return x_out

BATCH_SIZE = 100
LEARNING_RATE = 0.001
TOTAL_EPOCHS = 800
split_ratio = 0.05
change_learning_rate_epochs = 10

model_save = 'finetune_numble_10000_L2.pth'

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")

if __name__ == '__main__':

    # file_name1 = 'data/Case_3_Training.npy'
    file_name1 = 'data/Case_1_2_Training.npy'
    print('The current dataset is : %s' % (file_name1))
    CIR = np.load(file_name1)
    trainX = CIR.transpose((2, 1, 3, 0))  # [none, 256, 72, 2]
    trainX=trainX
    # file_name2 = 'data/Case_3_Training_Label.npy'
    file_name2 = 'data/Case_1_2_Training_Label.npy'
    print('The current dataset is : %s' % (file_name2))
    POS = np.load(file_name2)
    trainY = POS.transpose((1, 0))  # [none, 2]
    trainY=trainY
    # trainY=trainX

    # trainX=np.repeat(trainX,10,axis=0)
    # trainY=np.repeat(trainY,10,axis=0)

    # shuffle_idx = np.arange(trainY.shape[0])
    # np.random.shuffle(shuffle_idx)
    #
    # trainX = trainX[shuffle_idx]
    # trainY = trainY[shuffle_idx]


    model = ModelVAE()
    model_dict=model.state_dict()
    pre_dict=torch.load("pre_75_noise_0-20.pth")
    pre_dict={k:v for k,v in pre_dict.items() if k in model_dict }
    model_dict.update(pre_dict)
    model.load_state_dict(model_dict)

    # model.load_state_dict(torch.load("finetune_numble_3000_test_noisy_0-20_2.pth"))
    for params in model.encoder.parameters():
        params.requires_grad = False
    for params in model.fc_mu.parameters():
        params.requires_grad = False




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
    criterion = nn.L1Loss().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    test_avg_min = 10000;

    Train_log=[]
    Test_log=[]
    for epoch in range(TOTAL_EPOCHS):
        model.train()
        optimizer.param_groups[0]['lr'] = LEARNING_RATE / np.sqrt(np.sqrt(epoch + 1))

        # Learning rate decay
        if (epoch + 1) % change_learning_rate_epochs == 0:
            optimizer.param_groups[0]['lr'] /= 2
            print('lr:%.4e' % optimizer.param_groups[0]['lr'])
        # if (epoch + 6) % (change_learning_rate_epochs) == 0:
        #     optimizer.param_groups[0]['lr'] *= 2
            print('lr:%.4e' % optimizer.param_groups[0]['lr'])

        # Training in this epoch
        loss_avg = 0
        for i, (x, y) in enumerate(train_loader):
            x = x.float().to(DEVICE)
            patch=int (random.random()*15+3)
            #patch=8
            # mask=torch.hstack([torch.zeros(patch), torch.ones(18-patch)])

            #
            # if random.random()<0.6:
            #     mask = np.hstack([np.zeros(patch), np.ones(18 - patch)])
            #     np.random.shuffle(mask)
            #     mask = mask.repeat(4)
            #     mask = mask == 1
            #     x[:, :, ~mask, :] = 0

            #     for i in range(x.shape[0]):
            #         idx = torch.randperm(mask.nelement())
            #         mask = mask[idx]
            #         mask = mask == 1
            #         #     # p = mask.cuda()
            #         #     # p = torch.unsqueeze(p, axis=0)
            #         #     # # p = p.repeat([256,1])
            #         #     # p = torch.unsqueeze(p, axis=2)
            #         #     # # p = p.repeat([1,1,2])
            #         #     # p = torch.unsqueeze(p, axis=0)
            #         #     # p = p.repeat([1, 256, 1, 2])
            #         x[i, :, ~mask, :] = 0



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
        Train_log=np.append(Train_log,loss_avg)

        # Testing in this epoch
        model.eval()
        test_avg = 0
        for i, (x, y) in enumerate(test_loader):
            x = x.float().to(DEVICE)

            y = y.float().to(DEVICE)

            output = model(x)
            # 计算损失函数
            loss_test = criterion(output, y)
            test_avg += loss_test.item()

        test_avg /= len(test_loader)
        Test_log = np.append(Test_log, test_avg)
        print('Epoch : %d/%d, Loss: %.6f, Test: %.6f, BestTest: %.6f' % (
        epoch + 1, TOTAL_EPOCHS, loss_avg, test_avg, test_avg_min))
        if test_avg < test_avg_min:
            print('Model saved!')
            test_avg_min = test_avg

    np.save('train_loss_L1',Train_log)
    np.save('Test_loss_L1', Test_log)
else:
    print("load torch model")
    # model_ckpt = torch.load(model, model_save)
