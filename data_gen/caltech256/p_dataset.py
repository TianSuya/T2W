from torch.utils.data import Dataset, DataLoader
import json, clip, torch
from .normalization import get_normalizer

def flatten_pt(pt_data):
    flatten = []
    for key in pt_data.keys():
        flatten.append(pt_data[key].flatten())
    return torch.cat(flatten, dim=0)

MAX_VALUE = 0.150326207280159
MIN_VALUE = -0.1498246192932129

class PDataset(Dataset):

    def __init__(
        self, 
        config_path,
        normalizer_name: str = "openai",
        openai_coeff: float = 4.185
    ):
        super().__init__()
        with open(config_path, 'r') as f:
            self.data = json.load(f)
        self.parameter_sizes = [[16*512, 16],[512*16, 512]]
        self.parameter_names = ['weight', 'bias', 'weight', 'bias']
        self.normalizer_name = normalizer_name
        self.openai_coeff = openai_coeff
        self.min_val, self.max_val = MIN_VALUE, MAX_VALUE
        self.normalizer = get_normalizer(self.normalizer_name, openai_coeff=self.openai_coeff,
                                         min_val=self.min_val, max_val=self.max_val, dataset=self)
        
    def __getitem__(self, index):
        now_data = self.data[index]
        sd = torch.load(now_data['path'], map_location='cpu')
        # print(sd.keys())
        p_data = sd['model_state']
        p_data = flatten_pt(p_data)
        text_features = torch.tensor(now_data['text_features'])
        selected_classes = now_data['selected_classes']

        p_data = self.normalize(p_data)

        return {
            'p_data': p_data,
            'index': now_data['index'],
            'text': text_features
        }
    
    def __len__(self):
        return len(self.data)
    
    def normalize(self, weights):
        return self.normalizer.normalize(weights)

    def unnormalize(self, normalized_weights):
        return self.normalizer.unnormalize(normalized_weights)
    
if __name__ == '__main__':
    min_vals = 99
    max_vals = -99
    dataset = PDataset(
        config_path='./train.json',
    )
    for a in dataset:
        a = a['p_data']
        print(a)
        min_val = torch.min(a).item()
        max_val = torch.max(a).item()
        if min_val < min_vals: min_vals = min_val
        if max_val > max_vals: max_vals = max_val
    print(min_vals)
    print(max_vals)        