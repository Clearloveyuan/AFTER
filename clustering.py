# import argparse
import numpy as np
import torch
import argparse
from downstream.Custom.dataloader import CustomEmoDataset
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForXVector
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.features import RadViz
from utilities.general import similarity,get_nearest_to_centers
import os 
import json
import matplotlib.font_manager
import matplotlib
import random

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--max_epochs', type=int, default=20)
parser.add_argument('--maxseqlen', type=float, default=10)
parser.add_argument('--nworkers', type=int, default=4)
parser.add_argument('--precision', type=int, choices=[16, 32], default=32)
parser.add_argument('--saving_path', type=str, default='downstream/checkpoints/custom')
parser.add_argument('--datadir', type=str, required=True)
parser.add_argument('--labeldir', type=str, required=True)
parser.add_argument('--pretrained_path', type=str, default=None)
parser.add_argument('--model_type', type=str, choices=['wav2vec', 'wav2vec2'], default='wav2vec2')
parser.add_argument('--save_top_k', type=int, default=2)
parser.add_argument('--num_exps', type=int, default=5)
parser.add_argument('--outputfile', type=str, default=None)
parser.add_argument('--select_pro', type=float, default=0.1)
parser.add_argument('--init_train_data', type=int, default=200)
parser.add_argument('--save_name',type=str, default =None)
args = parser.parse_args()



"""
Clustering based initialization datasets    
"""

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
#breakpoint()
model = Wav2Vec2ForXVector.from_pretrained("anton-l/wav2vec2-base-superb-sv", output_hidden_states=True).to(device)
breakpoint()
dataset = CustomEmoDataset(args.datadir, args.labeldir, maxseqlen = args.maxseqlen)
breakpoint()
pool_loader = DataLoader(dataset=dataset.train_dataset,
                collate_fn=dataset.seqCollate,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.nworkers)
breakpoint()
sample = torch.zeros(1,512).to(device)
for batch in pool_loader:
    feats, length, label = batch
    feats = feats.to(model.device)
    with torch.no_grad():
        embeddings = model(feats).embeddings
        sample = torch.cat((sample, embeddings),0)
breakpoint()
sample=sample.to("cpu")
idx = 0
sample = sample[torch.arange(sample.size(0))!=0] 
np.save(args.save_name,sample)
breakpoint()
#breakpoint()


sample = np.load(args.save_name+".npy", allow_pickle=True)
#breakpoint()
#visualizer = RadViz(size=(756, 504))
#model = KMeans()
#visualizer = KElbowVisualizer(model, k=(1,8))
#breakpoint()
#visualizer.fit(sample)
breakpoint()
model = KElbowVisualizer(KMeans(), k=(1,8))
breakpoint()
model.fit(sample)
breakpoint()
print(model.elbow_value_)
   
#visualizer.show(outpath="Clustering.pdf")       
#breakpoint()