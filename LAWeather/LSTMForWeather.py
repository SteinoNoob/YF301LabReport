from sklearn.preprocessing import MinMaxScaler
from torch import Tensor, nn, optim
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

csv = pd.read_csv(r'./seattle-weather.csv', parse_dates=['date'], index_col=['date'])
features = ['precipitation', 'temp_max', 'temp_min', 'wind']

scaler = MinMaxScaler((0,1))
scaled_data = scaler.fit_transform(csv[features])
scaled_train = scaled_data[:1168]
scaled_test = scaled_data[1168:]


# hyp params
output_size = 7
batch_size = 64
base_lr = 0.001
sq_length = 30
input_dim = len(features)


class WeatherDataset(Dataset):
    def make_dataset(self, data, look_back, n_pred):
        data_number, data_label = [], []
        for i in range(len(data)-look_back-n_pred):
            data_number.append(data[i:i+look_back,:])
            data_label.append(data[i+look_back:i+look_back+n_pred,:])
            data_number = np.array(data_number)
            data_label = np.array(data_label)
            return torch.FloatTensor(data_number), torch.FloatTensor(data_label)
    def __init__(self, data, look_back, n_pred):
        super().__init__()
        self.x, self.label = self.make_dataset(data, look_back, n_pred)
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        return self.x[idx], self.label[idx]
    
train_set = WeatherDataset(scaled_train, 30, 7)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

class RNNPred(nn.Module):
    def __init__(self, input_dim, lstm_dim, dense_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, dense_dim//2),
            nn.ReLU(),
            nn.Linear(dense_dim//2, dense_dim),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_dim, lstm_dim, batch_first=True, bidirectional=True)
        self.logits = nn.Sequential(
            nn.Linear(lstm_dim*2, dense_dim),
            nn.ReLU(),
            nn.Linear(dense_dim, output_dim)
        )
    def forward(self, x):
        x = self.mlp(x)
        x, _ = self.lstm(x)
        x = self.logits(x)
        return x

model = RNNPred(input_dim, 128, 256, output_size).cuda()
optimer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=1e-5)
loss_fn = nn.HuberLoss()

losses = []
epochs = 100
if __name__ == '__main__':
    # for idx, (seq, l) in enumerate(weather_loader):
    #     print(seq.shape)
    #     if idx == 1:
    #         break
    model.train()
    for epoch in range(epochs):
        bar = tqdm(total=len(train_set),desc='进度',ncols=80)
        bar.set_description(f"epoch {epoch+1}/{epochs}")
        loss_value = 0
        for idx, (seq, l) in enumerate(train_loader):
            seq, l = seq.cuda(), l.cuda()
            out = model(seq)
            loss = loss_fn(out, l)
            optimer.zero_grad()
            loss.backward()
            optimer.step()
            loss_value = loss.item()
       
            bar.set_postfix(loss=f"{loss.item():.6f}")
            bar.update(batch_size)
        losses.append(loss_value)  
        bar.close()
    torch.save(model.state_dict(),"./weather_pred.pth")