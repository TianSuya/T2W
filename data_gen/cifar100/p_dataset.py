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
        self.parameter_sizes = [[16*512, 16], [512*16, 512]]  # Two-layer weight parameters
        self.parameter_names = ['weight', 'bias', 'weight', 'bias']
        self.normalizer_name = normalizer_name
        self.openai_coeff = openai_coeff
        self.min_val, self.max_val = MIN_VALUE, MAX_VALUE
        self.normalizer = get_normalizer(self.normalizer_name, openai_coeff=self.openai_coeff,
                                         min_val=self.min_val, max_val=self.max_val, dataset=self)
        
    def __getitem__(self, index):
        now_data = self.data[index]
        sd = torch.load(now_data['path'], map_location='cpu')
        
        # Process original weights
        original_weights = sd['clip_adapter']
        p_data = flatten_pt(original_weights)
        p_data = self.normalize(p_data)
        
        # Process permuted weights
        permuted_weights = self.permute_layer_weights(original_weights)
        p_data_permu = flatten_pt(permuted_weights)
        p_data_permu = self.normalize(p_data_permu)

        text_features = torch.tensor(now_data['text_features'])
        selected_classes = now_data['selected_classes']

        return {
            'p_data': p_data,
            'p_data_permu': p_data_permu,  # Permuted weights for augmentation
            'index': now_data['index'],
            'text': text_features
        }
    
    def __len__(self):
        return len(self.data)
    
    def normalize(self, weights):
        return self.normalizer.normalize(weights)

    def unnormalize(self, normalized_weights):
        return self.normalizer.unnormalize(normalized_weights)
    
    def permute_layer_weights(self, weights_dict):

        perm_dict = {k: v.clone() for k, v in weights_dict.items()}
        layer1_out = perm_dict['linear1.weight'].shape[0]
        
        torch.manual_seed(hash(str(perm_dict['linear1.weight'][:2])) % 2**32)
        perm = torch.randperm(layer1_out)
        
        perm_dict['linear1.weight'] = perm_dict['linear1.weight'][perm, :]
        perm_dict['linear1.bias'] = perm_dict['linear1.bias'][perm]
        perm_dict['linear2.weight'] = perm_dict['linear2.weight'][:, perm]
        
        return perm_dict
   