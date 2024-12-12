import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from molecule import Molecule
from rdkit import Chem
from rdkit.Chem import AllChem
import random
import pickle
import os
import glob
from multiprocessing import Process
import multiprocessing
from predict import predictNetworkDouble


MAXSETP = 40
ATOMTYPE = ['C', 'N', 'O']


def epsilon_value(epoch):
    if epoch < 50:
        epsilon = 1 - 0.9 / 50 * epoch
    else:
        epsilon = 0.1 - 0.09 / 50 * (epoch - 50)
    return epsilon


class HeuristicNetModule(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(HeuristicNetModule, self).__init__()
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

    def feature_and_value(self, x):
        for i in range(len(self.layers)):
            x = F.relu(self.layers[i][1](self.layers[i][0](x)))
        feature = F.sigmoid(x)
        out = self.fc_out(x)
        return out, feature

    def forward(self, x):
        for i in range(len(self.layers)):
            x = F.relu(self.layers[i][1](self.layers[i][0](x)))
        x = self.fc_out(x)
        return x


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


def flatten(data):
    num_each = [len(x) for x in data]
    split_idxs = list(np.cumsum(num_each)[:-1])
    data_flat = [item for sublist in data for item in sublist]
    return data_flat, split_idxs


def unflatten(data, split_idxs):
    data_split = []
    start_idx = 0
    for end_idx in split_idxs:
        data_split.append(data[start_idx: end_idx])
        start_idx = end_idx
    data_split.append(data[start_idx:])
    return data_split


def estimate_heuristic_value_Batch(model, fps, device):
    minibatch_size = 128
    num_batch = int(len(fps) // minibatch_size)
    if num_batch * minibatch_size < len(fps):
        num_batch += 1
    heuristic_values = np.array([])
    for i in range(num_batch):
        start = i * minibatch_size
        end = min((i + 1) * minibatch_size, len(fps))
        fps_tensor = torch.tensor(np.array(fps[start: end]), device=device)
        heuristic_values = np.append(heuristic_values, model(fps_tensor).cpu().data.numpy().reshape(-1))
    return heuristic_values


def samples_a_action(g_values, values, epoch):
    if np.random.uniform() > epsilon_value(epoch):
        f_values = np.array(g_values) + np.array(values)
        index = np.argsort(f_values)[-3:]
        return np.random.choice(index)
    else:
        return np.random.choice([i for i in range(len(values))])


def collect_games(num_games, device, epoch, thread):
    model = HeuristicNetModule(input_dim=2048, hidden_dims=[1024, 512, 128, 32]).to(device)
    parameters = torch.load('./model_novel/model_{:d}.model'.format(epoch), map_location={'cuda:0': device})
    model.load_state_dict(parameters)
    model.eval()
    envs = []
    data = [[] for _ in range(num_games)]
    for i in range(num_games):
        env = Molecule(
            atom_types=ATOMTYPE,
            allow_removal=True,
            allow_no_modification=True,
            allow_bonds_between_rings=True,
            allow_ring_sizes=[3, 4, 5, 6],
            max_steps=MAXSETP
        )
        env.initialize()
        envs.append(env)
    for step in range(MAXSETP):
        if step == MAXSETP - 1:
            for i in range(num_games):
                rewards = envs[i].batch_reward(envs[i]._valid_actions)
                move = np.argmax(rewards)
                result = envs[i].step(envs[i]._valid_actions[move])
                data[i].append([result.state, result.reward])
        else:
            fps = []
            batch_rewards = []
            for i in range(num_games):
                fps.append(batch_smiles_to_fingerprints(envs[i]._valid_actions))
                batch_rewards.append(envs[i].batch_reward(envs[i]._valid_actions))
            fps, split_idxs = flatten(fps)
            heuristic_values = estimate_heuristic_value_Batch(model, fps, device)
            heuristic_values = unflatten(heuristic_values, split_idxs)
            for i in range(num_games):
                move = samples_a_action(batch_rewards[i], heuristic_values[i], epoch)
                result = envs[i].step(envs[i]._valid_actions[move])
                data[i].append([result.state, result.reward])
    file_name = './data_novel/data_' + str(epoch) + '_' + str(thread) + '.pkl'
    with open(file_name, 'wb') as writer:
        pickle.dump(data, writer, protocol=4)


def cal_novelty(experts, smiles, device):
    fps = batch_smiles_to_fingerprints(smiles)
    fps = torch.tensor(fps, device=device)
    estimated_QEDs = []
    estimated_SAs = []
    for expert in experts:
        estimated_QED, estimated_SA = expert(fps)
        estimated_QEDs.append(estimated_QED.cpu().data.numpy().reshape(-1))
        estimated_SAs.append(estimated_SA.cpu().data.numpy().reshape(-1))
    estimated_QEDs, estimated_SAs = np.array(estimated_QEDs), np.array(estimated_SAs)
    SA_mean = 3.0525811293166134
    SA_std = 0.8335207024513095
    estimated_SAs = (estimated_SAs - SA_mean) / SA_std
    estimated_heuristics = estimated_QEDs - 0.2 * estimated_SAs
    novelty_reward = []
    for i in range(len(smiles)):
        novelty_reward.append(np.std(estimated_heuristics[:, i]))
    return novelty_reward


def prepare_data(epoch, device='cuda:0'):
    experts = []
    hidden_dims = [1024, 1024, 512, 512]
    QED_dims = [128, 128, 32, 32]
    SA_dims = [128, 128, 32, 32]
    input_dim = 2048
    for i in range(5):
        expert = predictNetworkDouble(input_dim, hidden_dims, QED_dims, SA_dims).to(device)
        parameters = torch.load('./train/model_{:d}/model_{:d}.model'.format(i, 48), map_location={'cuda:{:d}'.format(i % torch.cuda.device_count()): device})
        expert.load_state_dict(parameters)
        expert.eval()
        experts.append(expert)
    file_names = glob.glob('./data_novel/*.pkl')
    data = {
        'state': [],
        'target': []
    }
    for file in file_names:
        with open(file, 'rb') as f:
            current = pickle.load(f)
        generated_smiles = [path[-1][0] for path in current]
        novelty_reward = cal_novelty(experts, generated_smiles, device)
        for i in range(len(current)):
            data['state'] += [current[i][j][0] for j in range(len(current[i]))]
            data['target'] += [current[i][-1][1] - current[i][j][1] + 0.5 * novelty_reward[i] for j in range(len(current[i]))]
        os.remove(file)
    file_name = './data_saved_novel/data_{:d}.pkl'.format(epoch)
    with open(file_name, 'wb') as writer:
        pickle.dump(data, writer, protocol=4)


class HeuristicDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['target'])

    def __getitem__(self, item):
        return self.data['state'][item], self.data['target'][item]


def update(epoch, model, device, optimizer):
    file_name = './data_saved_novel/data_{:d}.pkl'.format(epoch)
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    os.remove(file_name)
    data['state'] = batch_smiles_to_fingerprints(data['state'])
    model.train()
    data = HeuristicDataSet(data)
    trainLoader = DataLoader(data, batch_size=1024, shuffle=True, num_workers=8)
    loss_fn = nn.MSELoss(reduction='mean')
    for batch in trainLoader:
        state, target = batch
        state = torch.tensor(np.array(state), device=device).float()
        target = torch.tensor(np.array(target).reshape(-1), device=device).float()
        predict_target = model(state)
        loss = loss_fn(predict_target.reshape(-1), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss.data.item()
        fr = open('heuristic_loss_novel.txt', 'a')
        fr.write(str(loss) + '\n')
        fr.close()
    torch.save(model.state_dict(), './model_novel/model_{:d}.model'.format(epoch + 1))
    return model, optimizer


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    device = 'cuda:0'
    model = HeuristicNetModule(input_dim=2048, hidden_dims=[1024, 512, 128, 32]).to(device)
    parameters = torch.load('./model_novel/model_{:d}.model'.format(0), map_location={'cuda:0': device})
    model.load_state_dict(parameters)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)
    for epoch in range(100):
        jobs = [Process(target=collect_games, args=(100, 'cuda:{:d}'.format(i % torch.cuda.device_count()), epoch, i)) for i in range(20)]
        for job in jobs:
            job.start()
        for job in jobs:
            job.join()
        prepare_data(epoch, device)
        model, optimizer = update(epoch, model, device, optimizer)





