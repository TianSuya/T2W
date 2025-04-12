import json
import os
import random
import torch
import clip
import tqdm

base_dir = '/data/bowen/Text2Weight/data_gen/cifar100/checkpoints'
data_range = [i for i in range(50000)]
dir_names = os.listdir(base_dir)
results = []
clip_model, _ = clip.load("ViT-B/32", device='cuda:4')

def split_list(lst, ratios, num_splits):
    """
    将列表按照指定比例和数量拆分成子列表
    :param lst: 待拆分列表
    :param ratios: 每个子列表的元素占比，由小数表示的列表
    :param num_splits: 子列表的数量
    :return: 拆分后的子列表组成的列表
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
    # 计算全局平均向量
    global_avg = torch.mean(embeddings, dim=0)

    # 计算余弦相似度（假设向量已L2归一化）
    embeddings_normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    global_avg_normalized = torch.nn.functional.normalize(global_avg.unsqueeze(0), p=2, dim=1)

    # 计算相似度得分
    scores = torch.mm(embeddings_normalized, global_avg_normalized.T).squeeze()
    weights = torch.softmax(scores, dim=0)

    # 加权平均
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
        info['path'] = file_path
        # p_data = torch.load(file_path, map_location='cuda:7')
        # print(p_data.keys())
        # if '' not in p_data.keys():continue
        selected_classes = info['selected_classes']
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in selected_classes])
        text_features = clip_model.encode_text(text_inputs.cuda(4))
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