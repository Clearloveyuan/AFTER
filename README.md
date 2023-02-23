# AFTER
Official implementation for the paper [Active Learning Based Fine-Tuning for Speech Emotion Recognition]


[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)
![MIT](https://img.shields.io/badge/license-MIT-yellowgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-orange)
![Release](https://img.shields.io/badge/release%20date-Oct%202022-blue)


## Dependencies
 - Python 3.9.1
 - Pytorch 1.9.0+cu111
 - Pytorch-lightning 1.7.4 
 - Transformers 4.10.0


## Install
To easily reproduce our results, you can install the environments by
```
pip install -r requirements.txt
```

## Prepare IEMOCAP
Obtain [IEMOCAP](https://sail.usc.edu/iemocap/) from USC
```
cd Dataset/IEMOCAP &&
python make_16k.py IEMOCAP_DIR &&
python gen_meta_label.py IEMOCAP_DIR &&
python generate_labels_sessionwise.py &&
cd ../..
```


## Usage

### Step One

Task Adaptation Pre-training process on IEMOCAP

```
bash bin/run_exp_iemocap_baseline.sh Dataset/IEMOCAP/Audio_16k/Dataset/IEMOCAP/labels_sess/label_2.json output_iemocap_train 2 TAPT 1
```

### Step Two

Calculate how many clusters in the dataset

```
CUDA_VISIBLE_DEVICES=2 python clustering.py --datadir Dataset/IEMOCAP/Audio_16k/ --labeldir Dataset/IEMOCAP/labels_original_sess/label_2.json --save_name iemocap_train
```

If you face the problem ImportError: cannot import name 'Wav2Vec2ForXVector' from 'transformers'
You could update your transformers version to **4.22.1**


### Step Three


* Run with **Random** as initialization for Active Learning method
```
bash bin/TAPT_full_train.sh Dataset/IEMOCAP/Audio_16k/ Dataset/IEMOCAP/labels_sess_new/label_2.json output_full_1 3 TAPT 1 random output_iemocap_train_1/last.ckpt Least_confidence
```

* Run with **Clustering** as initialization for Active Learning method
```
bash bin/TAPT_full_train.sh Dataset/IEMOCAP/Audio_16k/ Dataset/IEMOCAP/labels_sess_new/label_2.json output_full_1 3 TAPT 1 clustering output_iemocap_train_1/last.ckpt Entropy
```

#### Notions

The different Active Learning methods are listed here
<ul>
<li>Entropy</li>
<li>Least_confidence</li>
<li>alps</li>
<li>margin_confidence</li>
<li>bald</li>
<li>badge</li>
</ul>



## License

MIT LICENSE
