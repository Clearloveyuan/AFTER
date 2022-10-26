# import argparse
import numpy as np
import torch
from downstream.Custom.dataloader import CustomEmoDataset
from torch.utils.data import DataLoader
#from transformers import Wav2Vec2ForXVector
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.features import RadViz
from utilities.general import similarity,get_nearest_to_centers
import os 
import json
import matplotlib.font_manager
import matplotlib
import random

# parser = argparse.ArgumentParser()
# parser.add_argument('--batch_size', type=int, default=64)
# parser.add_argument('--lr', type=float, default=1e-4)
# parser.add_argument('--max_epochs', type=int, default=20)
# parser.add_argument('--maxseqlen', type=float, default=10)
# parser.add_argument('--nworkers', type=int, default=4)
# parser.add_argument('--precision', type=int, choices=[16, 32], default=32)
# parser.add_argument('--saving_path', type=str, default='downstream/checkpoints/custom')
# parser.add_argument('--datadir', type=str, required=True)
# parser.add_argument('--labeldir', type=str, required=True)
# parser.add_argument('--pretrained_path', type=str, default=None)
# parser.add_argument('--model_type', type=str, choices=['wav2vec', 'wav2vec2'], default='wav2vec2')
# parser.add_argument('--save_top_k', type=int, default=2)
# parser.add_argument('--num_exps', type=int, default=5)
# parser.add_argument('--outputfile', type=str, default=None)
# parser.add_argument('--select_pro', type=float, default=0.1)
# parser.add_argument('--init_train_data', type=int, default=200)
# args = parser.parse_args()



"""
Clustering based initialization datasets    
"""

# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# model = Wav2Vec2ForXVector.from_pretrained("anton-l/wav2vec2-base-superb-sv", output_hidden_states=True).to(device)
# dataset = CustomEmoDataset(args.datadir, args.labelpath, maxseqlen = args.maxseqlen)
# pool_loader = DataLoader(dataset=dataset.train_dataset,
#                 collate_fn=dataset.seqCollate,
#                 batch_size=args.batch_size,
#                 shuffle=False,
#                 num_workers=args.nworkers)

# sample = torch.zeros(1,512).to(device)
# for batch in pool_loader:
#     feats, length, label = batch
#     feats = feats.to(device)
#     with torch.no_grad():
#         embeddings = model(feats).embeddings
#         sample = torch.cat((sample, embeddings),0)
# sample=sample.to("cpu")
# idx = 0
# sample = sample[torch.arange(sample.size(0))!=0] 
# np.save("train_sample.npy",sample)
# breakpoint()


# sample = np.load("train_sample.npy", allow_pickle=True)
# breakpoint()
# visualizer = RadViz(size=(756, 504))
# model = KMeans()
# visualizer = KElbowVisualizer(model, k=(1,8))
# breakpoint()
# visualizer.fit(sample)   
# visualizer.show(outpath="Clustering.pdf")       
# breakpoint()


def init_clustering(args):
    train_embedding = np.load("train_sample.npy", allow_pickle=True)
    km = KMeans(n_clusters=3)
    km.fit(train_embedding)     
    indices = get_nearest_to_centers(km.cluster_centers_, train_embedding, normalized=False)
    while len(indices) < args.init_train_data:
        for i in range(km.cluster_centers_.shape[0]):
            sim = similarity(km.cluster_centers_[None, i], train_embedding, normalized=False)
            sim[0, indices] = -np.inf
            indices = np.append(indices, sim.argmax())
            if len(indices) == args.init_train_data:
                break
    count1 = 0
    count2 = 0
    select_new_data={"Train":{}, "Val":{}, "Test":{}}
    with open(args.labelpath, "r" ,encoding="utf-8") as f1:
        pool_new_data = json.load(f1)
    count =0
    delet_key =[]
    for key in pool_new_data["Train"]:
        if count in indices:
            select_new_data["Train"][key] = pool_new_data["Train"][key]
            delet_key.append(key)
            count += 1
            count1 +=1
        else:
            count += 1
            count2 +=1
    select_new_data["Val"] = pool_new_data["Val"]
    select_new_data["Test"] = pool_new_data["Test"]
    for key in delet_key:
        del pool_new_data["Train"][key]
    assert len(pool_new_data["Train"]) == count2
    assert len(select_new_data["Train"]) == count1
    save_path = os.path.join(args.labeldir, "initialize.json")
    with open(save_path, "w", encoding="utf-8") as f2:
        json.dump(select_new_data, f2,indent=2)
    with open(args.labelpath, "w",encoding="utf-8") as  f3:
        json.dump(pool_new_data,f3,indent=2)

def init_random(args):
    indices = random.sample(range(1,4446), 200)
    count1 = 0
    count2 = 0
    select_new_data={"Train":{}, "Val":{}, "Test":{}}
    with open(args.labelpath, "r" ,encoding="utf-8") as f1:
        pool_new_data = json.load(f1)
    count =0
    delet_key =[]
    for key in pool_new_data["Train"]:
        if count in indices:
            select_new_data["Train"][key] = pool_new_data["Train"][key]
            delet_key.append(key)
            count += 1
            count1 +=1
        else:
            count += 1
            count2 +=1
    select_new_data["Val"] = pool_new_data["Val"]
    select_new_data["Test"] = pool_new_data["Test"]
    for key in delet_key:
        del pool_new_data["Train"][key]
    assert len(pool_new_data["Train"]) == count2
    assert len(select_new_data["Train"]) == count1
    save_path = os.path.join(args.labeldir, "initialize.json")
    with open(save_path, "w", encoding="utf-8") as f2:
        json.dump(select_new_data, f2,indent=2)
    with open(args.labelpath, "w",encoding="utf-8") as  f3:
        json.dump(pool_new_data,f3,indent=2)