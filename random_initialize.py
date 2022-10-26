import argparse
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




#首先需要讲labels_seess 中的label_2.json 复制到labels_new_seess重的label_2.json中
curpath = "Dataset/IEMOCAP/labels_sess/label_2.json"
labelpath = "Dataset/IEMOCAP/labels_sess_new/label_2.json"

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







indices = random.sample(range(1,4446), 200)
#breakpoint()
count1 = 0
count2 = 0

# hparams = args
# foldlabel = os.listdir(args.labeldir)[0]
# hparams.labelpath = os.path.join(hparams.labeldir, foldlabel)
select_new_data={"Train":{}, "Val":{}, "Test":{}}
#breakpoint()
with open(curpath, "r" ,encoding="utf-8") as f1:
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

#breakpoint()
assert len(pool_new_data["Train"]) == count2
assert len(select_new_data["Train"]) == count1

#breakpoint()
with open("bald_random/initialize.json", "w", encoding="utf-8") as f2:
    json.dump(select_new_data, f2,indent=2)

with open(labelpath, "w",encoding="utf-8") as  f3:
    json.dump(pool_new_data,f3,indent=2)
