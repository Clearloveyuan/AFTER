import os
import argparse
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.outputlib import WriteConfusionSeaborn
from downstream.Custom.trainer import DownstreamGeneral
from downstream.Custom.dataloader import CustomEmoDataset
from torch.utils.data import DataLoader
import json
from initialize import init_clustering, init_random


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--max_epochs', type=int, default=30)
parser.add_argument('--maxseqlen', type=float, default=10)
parser.add_argument('--nworkers', type=int, default=1)
parser.add_argument('--precision', type=int, choices=[16, 32], default=32)
parser.add_argument('--saving_path', type=str, default='downstream/checkpoints/custom')
parser.add_argument('--datadir', type=str, required=True)
parser.add_argument('--labeldir', type=str, required=True)
parser.add_argument('--pretrained_path', type=str, default=None)
parser.add_argument('--model_type', type=str, choices=['wav2vec', 'wav2vec2'], default='wav2vec2')
parser.add_argument('--save_top_k', type=int, default=1)
parser.add_argument('--num_exps', type=int, default=1)
parser.add_argument('--outputfile', type=str, default=None)
parser.add_argument('--inits_method', type=str, default=None)
parser.add_argument('--select_pro', type=list, default=[0.05,0.05,0.05,0.05])
parser.add_argument('--init_train_data', type=int, default=200)
parser.add_argument('--labelpath',type=str, default=None)
parser.add_argument('--active_learning',type=str, default=None)
args = parser.parse_args()


hparams = args
if not os.path.exists(hparams.saving_path):
    os.makedirs(hparams.saving_path)
nfolds = len(os.listdir(hparams.labeldir))

for foldlabel in os.listdir(hparams.labeldir):
    assert foldlabel[-5:] == '.json'

metrics, confusion = np.zeros((4, args.num_exps, nfolds)), 0.


"""
foldlabel = os.listdir(args.labeldir)[0]
args.labelpath = os.path.join(args.labeldir, foldlabel)
breakpoint()
labelpath_new = "play/initialize.json"
breakpoint()

#Clustering based initialization datasets    


#Here adopting elbow method to check the number of clusters


# args.labelpath = "OUTPUT4/labels/label_new_2.json"
# model = Wav2Vec2ForXVector.from_pretrained("anton-l/wav2vec2-base-superb-sv", output_hidden_states=True)
# dataset = CustomEmoDataset(args.datadir, args.labelpath, maxseqlen = args.maxseqlen)
# pool_loader = DataLoader(dataset=dataset.train_dataset,
#                 collate_fn=dataset.seqCollate,
#                 batch_size=args.batch_size,
#                 shuffle=False,
#                 num_workers=args.nworkers)
# sample =[]
# for batch in pool_loader:
#     feats, length, label = batch
#     with torch.no_grad():
#         embeddings = model(feats).embeddings
#         sample.append(embeddings)
# sample = np.array(sample)
# np.save("train_sample.npy",sample)
# visualizer = RadViz(size=(756, 504))
# model = KMeans()
# visualizer = KElbowVisualizer(model, k=(1,8))
# visualizer.fit(sample)
# visualizer.show(outpath="clustering.pdf")

#Here calculating the most cloest datasets of clustering center


# train_embedding = np.load("train_sample.npy", allow_pickle=True)
# km = KMeans(n_clusters=3)
# km.fit(train_embedding)     
# indices = get_nearest_to_centers(km.cluster_centers_, train_embedding, normalized=False)
# while len(indices) < args.init_train_data:
#     for i in range(km.cluster_centers_.shape[0]):
#         sim = similarity(km.cluster_centers_[None, i], train_embedding, normalized=False)
#         sim[0, indices] = -np.inf
#         indices = np.append(indices, sim.argmax())
#         if len(indices) == args.init_train_data:
#             break
        

#Creat Initialization 

# foldlabel = os.listdir(args.labeldir)[0]
# hparams.labelpath = os.path.join(hparams.labeldir, foldlabel)
# select_new_data={"Train":{}, "Val":{}, "Test":{}}
# with open(hparams.labelpath, "r" ,encoding="utf-8") as f1:
#     pool_new_data = json.load(f1)
# count =0
# delet_key =[]
# for key in pool_new_data["Train"]:

#     if count in indices:
#         select_new_data["Train"][key] = pool_new_data["Train"][key]
#         delet_key.append(key)
#         count += 1
#     else:
#         count += 1

# select_new_data["Val"] = pool_new_data["Val"]
# select_new_data["Test"] = pool_new_data["Test"]
# for key in delet_key:
#     del pool_new_data["Train"][key]

# with open("play_conf/initialize.json", "w", encoding="utf-8") as f3:
#     json.dump(select_new_data, f3,indent=2)

# with open(hparams.labelpath, "w",encoding="utf-8") as  f2:
#     json.dump(pool_new_data,f2,indent=2)

# breakpoint()

"""


foldlabel = os.listdir(args.labeldir)[0]
args.labelpath = os.path.join(args.labeldir, foldlabel)

if args.inits_method == "clustering":
    init_clustering(args)
else:
    init_random(args)


foldlabel = "initialize.json"
hparams.labelpath = os.path.join(args.labeldir, foldlabel)

model = DownstreamGeneral(hparams)

checkpoint_callback = ModelCheckpoint(
    dirpath= os.path.join(hparams.saving_path, str(5)+ "%"),
    filename='{epoch:02d}-{valid_loss:.3f}-{valid_UAR:.5f}' if hasattr(model, 'valid_met') else None,
    save_top_k=args.save_top_k,
    verbose=True,
    save_weights_only=False,
    monitor='valid_UAR' if hasattr(model, 'valid_met') else None,
    mode='max'
)

trainer = Trainer(
    precision=args.precision,
    amp_backend='native',
    resume_from_checkpoint=None,
    #callbacks=[checkpoint_callback] if hasattr(model, 'valid_met') else None,
    callbacks=None,
    check_val_every_n_epoch=1,
    max_epochs=hparams.max_epochs,
    num_sanity_val_steps=2 if hasattr(model, 'valid_met') else 0,
    gpus=1,
    logger=False
)

trainer.fit(model)

if hasattr(model, 'valid_met'):
    trainer.test()
else:
    trainer.test(model)
met = model.test_met
metrics[:,0,0] = np.array([met.uar*100, met.war*100, met.macroF1*100, met.microF1*100])
confusion += met.m
        
outputstr = f"+++ SUMMARY 初始化 +++\n"
for nm, metric in zip(('UAR [%]', 'WAR [%]', 'macroF1 [%]', 'microF1 [%]'), metrics):
    outputstr += f"Mean {nm}: {np.mean(metric):.2f}\n"
    outputstr += f"Fold Std. {nm}: {np.mean(np.std(metric, 1)):.2f}\n"
    outputstr += f"Fold Median {nm}: {np.mean(np.median(metric, 1)):.2f}\n"
    outputstr += f"Run Std. {nm}: {np.std(np.mean(metric, 1)):.2f}\n"
    outputstr += f"Run Median {nm}: {np.median(np.mean(metric, 1)):.2f}\n"
if args.outputfile:
    with open(args.outputfile, 'w') as f:
        f.write(outputstr)
        f.write("\n \n \n")
else:
    print (outputstr)        


print("*"*50)        
print("The first time to extract samples by active learning method")
print("*"*50)

foldlabel = "label_2.json"
args.labelpath = os.path.join(args.labeldir, foldlabel)

print("Incoporate dataset")
dataset = CustomEmoDataset(args.datadir, args.labelpath, args.maxseqlen)

pool_loader =  DataLoader(dataset=dataset.train_dataset,
                    collate_fn=dataset.seqCollate,
                    batch_size=1,
                    shuffle=False,
                    num_workers=4)

scores = trainer.predict(model, dataloaders=pool_loader,return_predictions=True)
number_pooldata = pool_loader.__len__()
select_sample = int(number_pooldata * 0.05)
sample_ids = np.argpartition(scores,select_sample)[-select_sample:]
sample_ids = np.sort(sample_ids)
select_data={}
print("Get all scores")
print("Open")
with open(args.labelpath,"r",encoding="utf-8") as f1:
    pool_data = json.load(f1)
count =0
delet_key =[]
for key in pool_data["Train"]:
    if count in sample_ids:
        select_data[key] = pool_data["Train"][key]
        delet_key.append(key)
        count += 1
    else:
        count += 1
for key in delet_key:
    del pool_data["Train"][key]
save_path = os.path.join(args.labeldir, "initialize.json")

with open(save_path, "r", encoding="utf-8") as f2:
    old_data = json.load(f2)
old_data['Train'].update(select_data)

with open(args.labelpath, "w",encoding="utf-8") as  f3:
    json.dump(pool_data,f3,indent=2)
with open(save_path, "w", encoding="utf-8") as f4:
    json.dump(old_data, f4,indent=2)
print("Finish writing the new file")











for i in range(len(args.select_pro)):
    print(f"Current select the {args.select_pro[i] *100}\% samples")
    hparams.labelpath = save_path
    #breakpoint()
    model = DownstreamGeneral(hparams)
    new_path = os.path.join(hparams.saving_path, str(5*(i+2))+ "%")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=new_path,
        filename='{args.select_pro}-{epoch:02d}-{valid_loss:.3f}-{valid_UAR:.5f}' if hasattr(model, 'valid_met') else None,
        save_top_k=args.save_top_k if hasattr(model, 'valid_met') else 0,
        verbose=True,
        save_weights_only=False,
        monitor='valid_UAR' if hasattr(model, 'valid_met') else None,
        mode='max'
    )
    original = os.path.join(hparams.saving_path, str(5*(i+1))+ "%")
    #breakpoint()

    trainer = Trainer(
        precision=args.precision,
        amp_backend='native',
        # callbacks=[checkpoint_callback] if hasattr(model, 'valid_met') else None,
        #resume_from_checkpoint= os.path.join(original,os.listdir(original)[0]),
        check_val_every_n_epoch=1,
        max_epochs=hparams.max_epochs,
        num_sanity_val_steps=2 if hasattr(model, 'valid_met') else 0,
        gpus=1,
        logger=False
    )
    trainer.fit(model)

    if hasattr(model, 'valid_met'):
        trainer.test()
    else:
        trainer.test(model)
    met = model.test_met
    metrics[:, 0, 0] = np.array([met.uar*100, met.war*100, met.macroF1*100, met.microF1*100])
    confusion += met.m

    outputstr = "+++ SUMMARY" +str(5*(i+1)) + "+++\n"
    for nm, metric in zip(('UAR [%]', 'WAR [%]', 'macroF1 [%]', 'microF1 [%]'), metrics):
        outputstr += f"Mean {nm}: {np.mean(metric):.2f}\n"
        outputstr += f"Fold Std. {nm}: {np.mean(np.std(metric, 1)):.2f}\n"
        outputstr += f"Fold Median {nm}: {np.mean(np.median(metric, 1)):.2f}\n"
        outputstr += f"Run Std. {nm}: {np.std(np.mean(metric, 1)):.2f}\n"
        outputstr += f"Run Median {nm}: {np.median(np.mean(metric, 1)):.2f}\n"
    if args.outputfile:
        with open(args.outputfile, 'a') as f:
            f.write(outputstr)
            f.write("\n \n \n")
    else:
        print (outputstr)     

    """ 
    Here adopting Active Learning method to select samples and make new datasets.
    """
    #breakpoint()
    foldlabel = "label_2.json"
    args.labelpath = os.path.join(args.labeldir, foldlabel)
    #breakpoint()
    
    print("Incoporate dataset")
    dataset = CustomEmoDataset(args.datadir, args.labelpath, args.maxseqlen)

    pool_loader =  DataLoader(dataset=dataset.train_dataset,
                        collate_fn=dataset.seqCollate,
                        batch_size=1,
                        shuffle=False,
                        num_workers=4)

    scores = trainer.predict(model, dataloaders=pool_loader,return_predictions=True)
    #number_pooldata = pool_loader.__len__()
    select_sample = int(4508 * args.select_pro[i])
    sample_ids = np.argpartition(scores,select_sample)[-select_sample:]
    sample_ids = np.sort(sample_ids)
    select_data={}
    print("Get all scores")
    print("Open")
    #breakpoint()
    with open(args.labelpath,"r",encoding="utf-8") as f1:
        pool_data = json.load(f1)

    count =0
    delet_key =[]
    for key in pool_data["Train"]:
        if count in sample_ids:
            select_data[key] = pool_data["Train"][key]
            delet_key.append(key)
            count += 1
        else:
            count += 1

    for key in delet_key:
        del pool_data["Train"][key]
    #breakpoint()
    with open(save_path, "r", encoding="utf-8") as f2:
        old_data = json.load(f2)
    old_data['Train'].update(select_data)
    #breakpoint()
    with open(args.labelpath, "w",encoding="utf-8") as  f3:
        json.dump(pool_data,f3,indent=2)
    #cur_path = "bald/initialize_"+str(i+2)+".json"
    #breakpoint()
    with open(save_path, "w", encoding="utf-8") as f4:
        json.dump(old_data, f4,indent=2)
    print("Finish writing the new file")
WriteConfusionSeaborn(
    confusion,
    model.dataset.emoset,
    os.path.join(args.saving_path, 'confmat.png')
)