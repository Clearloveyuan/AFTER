# AFTER
Official implementation for the paper [Active Learning Based Fine-Tuning for Speech Emotion Recognition]

Submitted to ICASSP 2023.

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

## Usage

### Step One

Task Adaptation Pre-training process on IEMOCAP

```
bash bin/run_exp_iemocap_baseline.sh Dataset/IEMOCAP/Audio_16k/Dataset/IEMOCAP/labels_sess/label_2.json output_iemocap_train 2 TAPT 1
```

### Step Two

Calculate how many clusters in the dataset

```
python clustering.py --datadir Dataset/IEMOCAP/Audio_16k/ --labeldir Dataset/IEMOCAP/Audio_16k/Dataset/IEMOCAP/labels_sess/label_2.json
```

### Step Three

```
bash bin/run_exp_iemocap_vft.sh Dataset/IEMOCAP/Audio_16k/ Dataset/IEMOCAP/labels_sess_new/label_2.json output_c_margin 1 V-FT 1 clustering
```

#### Notions

The different Active Learning methods can be chenged here



## Contributing
This project exists thanks to all the people who contribute.

<a href="[https://github.com/wykst]"> <img src="pics/profile/wang.png"  width="80" >  </a> 



## Thanks
We thanks xxx et al. for providing automativcally pretrained speech recognition [Wav2vec2.0](https://huggingface.co/docs/transformers/model_doc/wav2vec2) model for us.

We thanks William Falcon et al. for their [Pytorch-Lighting Tool](https://www.pytorchlightning.ai/team)

We thanks the framework proposed by Li-Wei Chen et al. and [their work](https://arxiv.org/pdf/2110.06309.pdf).

## License

[MIT](LICENSE) © Richard Littauer
