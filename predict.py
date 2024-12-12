import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
import pickle


class predictNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(predictNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims)):
            if i == 0:
                block_fc = nn.Linear(self.input_dim, self.hidden_dims[i])
            else:
                block_fc = nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i])
            block_bn = nn.BatchNorm1d(self.hidden_dims[i])
            self.layers.append(nn.ModuleList([block_fc, block_bn]))
        self.fc_out = nn.Linear(self.hidden_dims[-1], 1)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = F.relu(self.layers[i][1](self.layers[i][0](x)))
        x = F.relu(self.fc_out(x))
        return x


class predictNetworkDouble(nn.Module):
    def __init__(self, input_dim, hidden_dims, QED_dims, SA_dims):
        super(predictNetworkDouble, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.QED_dims =QED_dims
        self.SA_dims = SA_dims
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims)):
            if i == 0:
                block_fc = nn.Linear(self.input_dim, self.hidden_dims[i])
            else:
                block_fc = nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i])
            block_bn = nn.BatchNorm1d(self.hidden_dims[i])
            self.layers.append(nn.ModuleList([block_fc, block_bn]))
        self.QED_layers = nn.ModuleList()
        for i in range(len(QED_dims)):
            if i == 0:
                block_fc = nn.Linear(self.hidden_dims[-1], self.QED_dims[i])
            else:
                block_fc = nn.Linear(self.QED_dims[i - 1], self.QED_dims[i])
            block_bn = nn.BatchNorm1d(self.QED_dims[i])
            self.QED_layers.append(nn.ModuleList([block_fc, block_bn]))
        self.SA_layers = nn.ModuleList()
        for i in range(len(SA_dims)):
            if i == 0:
                block_fc = nn.Linear(self.hidden_dims[-1], self.SA_dims[i])
            else:
                block_fc = nn.Linear(self.SA_dims[i - 1], self.SA_dims[i])
            block_bn = nn.BatchNorm1d(self.SA_dims[i])
            self.SA_layers.append(nn.ModuleList([block_fc, block_bn]))
        self.QED_fc_out = nn.Linear(self.QED_dims[-1], 1)
        self.SA_fc_out = nn.Linear(self.SA_dims[-1], 1)
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.shape) < 2:
                torch.nn.init.xavier_normal_(weight.unsqueeze(0), gain=10)
            else:
                torch.nn.init.xavier_normal_(weight, gain=10)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = F.relu(self.layers[i][1](self.layers[i][0](x)))
        qed = x
        for i in range(len(self.QED_layers)):
            qed = F.relu(self.QED_layers[i][1](self.QED_layers[i][0](qed)))
        qed = F.relu(self.QED_fc_out(qed))
        sa = x
        for i in range(len(self.SA_layers)):
            sa = F.relu(self.SA_layers[i][1](self.SA_layers[i][0](sa)))
        sa = F.relu(self.SA_fc_out(sa))
        return qed, sa


class PredictDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['target'])

    def __getitem__(self, item):
        smiles = self.data['smiles'][item]
        target = self.data['target'][item]
        return smiles, target


class PredictDataSetDouble(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['target'])

    def __getitem__(self, item):
        smiles = self.data['smiles'][item]
        QED = self.data['target'][item][0]
        SA = self.data['target'][item][1]
        return smiles, QED, SA


def batch_smiles_to_fingerprints(smiles):
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        onbits = list(fp.GetOnBits())
        arr = np.zeros(fp.GetNumBits(), dtype=np.float32)
        arr[onbits] = 1
        fps.append(arr)
    return np.array(fps)


def test(model, test_data, device):
    model.eval()
    num_data = len(test_data['target'])
    test_data = PredictDataSetDouble(test_data)
    test_data_loader = DataLoader(test_data, batch_size=1024, shuffle=False, num_workers=8)
    QED_loss_fn = nn.MSELoss(reduction='sum')
    QED_loss_sum = 0.0
    SA_loss_fn = nn.MSELoss(reduction='sum')
    SA_loss_sum = 0.0
    for batch in test_data_loader:
        smiles, QED, SA = batch
        smiles_fp = batch_smiles_to_fingerprints(smiles)
        smiles_fp = torch.tensor(smiles_fp, device=device).float()
        QED = torch.tensor(np.array(QED).reshape(-1), device=device).float()
        SA = torch.tensor(np.array(SA).reshape(-1), device=device).float()
        predict_QED, predict_SA = model(smiles_fp)
        QED_loss = QED_loss_fn(predict_QED.reshape(-1), QED)
        QED_loss_sum += QED_loss.data.item()
        SA_loss = SA_loss_fn(predict_SA.reshape(-1), SA)
        SA_loss_sum += SA_loss.data.item()
    return QED_loss_sum / num_data, SA_loss_sum / num_data


if __name__ == "__main__":
    round = 0
    hidden_dims = [1024, 1024, 512, 512]
    QED_dims = [128, 128, 32, 32]
    SA_dims = [128, 128, 32, 32]
    input_dim = 2048
    device = 'cuda:{:d}'.format(round % torch.cuda.device_count())
    model = predictNetworkDouble(input_dim=input_dim, hidden_dims=hidden_dims, QED_dims=QED_dims, SA_dims=SA_dims).to(device)
    modelName = './train/model_{:d}/model_{:d}.model'.format(round, 0)
    torch.save(model.state_dict(), modelName)
    train_index = [i for i in range(5)]
    train_data_smiles = []
    train_data_target = []
    for index in train_index:
        file = './ChEMBL_data/ChemBL_QED_SA_{:d}.pkl'.format(index)
        with open(file, 'rb') as f:
            data = pickle.load(f)
            train_data_smiles += [data[i][0] for i in range(len(data))]
            train_data_target += [[data[i][1], data[i][2]] for i in range(len(data))]
    train_data = {
        'smiles': train_data_smiles,
        'target': train_data_target
    }
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0000001)
    train_data = PredictDataSetDouble(train_data)
    train_data_loader = DataLoader(train_data, batch_size=1024, shuffle=True, num_workers=8)
    QED_loss_fn = nn.MSELoss(reduction='mean')
    SA_loss_fn = nn.MSELoss(reduction='mean')
    for epoch in range(100):
        model.train()
        for batch in train_data_loader:
            smiles, QED, SA = batch
            smiles_fp = batch_smiles_to_fingerprints(smiles)
            smiles_fp = torch.tensor(smiles_fp, device=device).float()
            QED = torch.tensor(np.array(QED).reshape(-1), device=device).float()
            SA = torch.tensor(np.array(SA).reshape(-1), device=device).float()
            predict_QED, predict_SA = model(smiles_fp)
            QED_loss = QED_loss_fn(predict_QED.reshape(-1), QED)
            SA_loss = SA_loss_fn(predict_SA.reshape(-1), SA)
            loss = QED_loss + SA_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.data.item()
            QED_loss = QED_loss.data.item()
            SA_loss = SA_loss.data.item()
            fr = open('./train/loss_{:d}.txt'.format(round), 'a')
            fr.write(str(loss) + '\t' + str(QED_loss) + '\t' + str(SA_loss) + '\n')
            fr.close()
        modelName = './train/model_{:d}/model_{:d}.model'.format(round, epoch + 1)
        torch.save(model.state_dict(), modelName)



