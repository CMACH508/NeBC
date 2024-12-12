import pickle
from heuristic import HeuristicNetModule
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from molecule import Molecule
from sklearn.cluster import KMeans


class BeamSearch:
    def __init__(self, env, nnet, device, beamsize, num_cluster=5):
        self.env = env
        self.nnet = nnet
        self.device = device
        self.beam_size = beamsize
        self.open = self.env.atom_types
        self.beam_size = beamsize
        self.num_cluster = num_cluster
        self.num_per_cluster = int(self.beam_size / self.num_cluster)

    def batch_smiles_to_fingerprints(self, smiles):
        fps = []
        for s in smiles:
            mol = Chem.MolFromSmiles(s)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            onbits = list(fp.GetOnBits())
            arr = np.zeros(fp.GetNumBits(), dtype=np.float32)
            arr[onbits] = 1
            fps.append(arr)
        return np.array(fps)

    def estimate_Q_value(self, fps):
        fps_tensor = torch.tensor(fps, device=self.device)
        Q_values, Q_features = self.nnet.feature_and_value(fps_tensor)
        Q_values = Q_values.cpu().data.numpy().reshape(-1)
        Q_features = Q_features.cpu().data.numpy().reshape(-1, 32)
        return Q_values, Q_features

    def heuristic_fn(self, states):
        fps = self.batch_smiles_to_fingerprints(states)
        estimated_Qs, estimated_features = self.estimate_Q_value(fps)
        return estimated_Qs, estimated_features

    def step(self):
        child_states = []
        for state in self.open:
            child_states += list(self.env.get_valid_actions(state))
        child_states = list(set(child_states))
        if len(child_states) > self.beam_size:
            batch_size = 1024000
            num_batch = len(child_states) // batch_size
            if num_batch * batch_size < len(child_states):
                num_batch += 1
            path_costs = []
            heuristics = []
            features = []
            for i in range(num_batch):
                start = i * batch_size
                end = min((i + 1) * batch_size, len(child_states))
                path_costs += list(self.env.batch_reward(child_states[start: end]))
                heuristic, feature = self.heuristic_fn(child_states[start: end])
                heuristics += list(heuristic)
                features += list(feature)
            kmeans = KMeans(n_clusters=self.num_cluster, n_init=1)
            kmeans.fit(features)
            y_kmeans = kmeans.predict(features)
            cost = np.array(path_costs) + np.array(heuristics)
            moves = []
            for i in range(self.num_cluster):
                index_cluster = [j for j in range(len(y_kmeans)) if y_kmeans[j] == i]
                if len(index_cluster) <= self.num_per_cluster:
                    index_candidate = index_cluster
                else:
                    cost_candidate = [cost[j] for j in index_cluster]
                    index_candidate = [index_cluster[j] for j in np.argsort(cost_candidate)[-self.num_per_cluster:]]
                moves += index_candidate
            child_states = [child_states[move] for move in moves]
        self.open = child_states

    def search(self):
        for i in range(38):
            print(i)
            self.step()
        child_states = []
        for state in self.open:
            child_states += list(self.env.get_valid_actions(state))
        child_states = list(set(child_states))
        path_costs = list(self.env.batch_reward(child_states))
        moves = np.argsort(path_costs)[-1000:]
        child_states = [child_states[move] for move in moves]
        cost = [path_costs[move] for move in moves]
        ans = {
            'smiles': child_states,
            'score': cost
        }
        with open('./test_result/result_beam_cluster_1000.pkl', 'wb') as writer:
            pickle.dump(ans, writer, protocol=4)


if __name__ == "__main__":
    device = 'cuda:0'
    hidden_dims = [1024, 512, 128, 32]
    input_dim = 2048
    env = Molecule(
        init_mol=None,
        atom_types=['C', 'N', 'O'],
        allow_removal=True,
        allow_no_modification=True,
        allow_bonds_between_rings=True,
        allow_ring_sizes=[3, 4, 5, 6],
        max_steps=40
    )
    for beamsize in [100]:
        model = HeuristicNetModule(input_dim=2048, hidden_dims=[1024, 512, 128, 32]).to(device)
        parameters = torch.load('./model_novel/model.model', map_location={'cuda:0': device})
        model.load_state_dict(parameters)
        model.eval()
        player = BeamSearch(env, model, device, beamsize, num_cluster=10)
        player.search()
        
