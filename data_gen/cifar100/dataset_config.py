import json
import os
import random
import torch
import clip
import tqdm

# Configuration: modify these paths according to your setup
base_dir = './checkpoints'
device = 'cuda:0'

data_range = [i for i in range(50000)]
dir_names = os.listdir(base_dir)
results = []
clip_model, _ = clip.load("ViT-B/32", device=device)

def split_list(lst, ratios, num_splits):
    """
    Split a list into sublists according to specified ratios.
    :param lst: List to be split
    :param ratios: Ratio for each sublist, represented as decimals
    :param num_splits: Number of sublists
    :return: List of sublists
    """
    if len(ratios) != num_splits:
        raise ValueError("The length of ratios must equal to num_splits.")
    total_ratio = sum(ratios)
    if total_ratio != 1:
        raise ValueError("The sum of ratios must be equal to 1.")
    n = len(lst)
    result = []
    start = 0
    for i in range(num_splits):
        end = start + int(n * ratios[i])
        result.append(lst[start:end])
        start = end
    return result

def fuse_features(embeddings):
    # Compute global average vector
    global_avg = torch.mean(embeddings, dim=0)

    # Compute cosine similarity (assuming vectors are L2-normalized)
    embeddings_normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    global_avg_normalized = torch.nn.functional.normalize(global_avg.unsqueeze(0), p=2, dim=1)

    # Compute similarity scores
    scores = torch.mm(embeddings_normalized, global_avg_normalized.T).squeeze()
    weights = torch.softmax(scores, dim=0)

    # Weighted average
    fused_vector = torch.sum(embeddings * weights.unsqueeze(1), dim=0)
    return fused_vector


def find_value(dic, index):
    for item in dic:
        if item['index'] == index:
            return item

with open('data_point.json', 'r') as f:
    data_point = json.load(f)

for item in tqdm.tqdm(dir_names):
    file_path = os.path.join(base_dir, item, 'task.pt')
    if os.path.exists(file_path):
        info = find_value(data_point, int(item))
        if info is None:
            continue
        info['path'] = file_path
        selected_classes = info['selected_classes']
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in selected_classes])
        text_features = clip_model.encode_text(text_inputs.to(device))
        text_features = fuse_features(text_features)
        info['text_features'] = text_features.tolist()
        results.append(info)

random.seed(42)
random.shuffle(results)
train_dataset, val_dataset = split_list(results, [0.8, 0.2], 2)
print(len(train_dataset), len(val_dataset))

with open('train.json','w') as f:
    json.dump(train_dataset, f, indent=4)

with open('val.json','w') as f:
    json.dump(val_dataset, f, indent=4)