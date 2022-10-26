#!/bin/bash
if [ -d "$3" ]; then
    rm -r $3
fi
rm Dataset/IEMOCAP/labels_sess_new/label_2.json
cp Dataset/IEMOCAP/labels_sess/label_2.json Dataset/IEMOCAP/labels_sess_new/
mkdir -p $3/labels
cp $2 $3/labels/
w2v2_path="output_iemocap_train/last.ckpt";
CUDA_VISIBLE_DEVICES=$4 python run_downstream_custom_multiple_fold.py --precision 16 --num_exps $6 --datadir $1 --labeldir $3/labels --pretrained_path $w2v2_path --outputfile $3/$5.log --inits_method $7
