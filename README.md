# Dense Relational Captioning

The code for our [CVPR 2019](https://cvpr2019.thecvf.com/) paper,

**[Dense Relational Captioning: Triple-Stream Networks for Relationship-Based Captioning](https://sites.google.com/view/relcap)**.

Done by Dong-Jin Kim, Jinsoo Choi, Tae-Hyun Oh, and In So Kweon.

Link: **[arXiv](https://arxiv.org/pdf/1903.05942.pdf)** , **[Dataset](https://drive.google.com/file/d/1cCN36poslxe7cCMkLnhYK0a-Y3vO4Rfn/view?usp=sharing)**, **[Sample model](https://drive.google.com/file/d/19t6Ogcl_ZlW9G6sPLBiWXfepWlX7MXg3/view?usp=sharing)**.

(The instruction will be available soon.)

We introduce “relational captioning,” a novel image captioning task which aims to generate multiple captions with respect to relational information between objects in an image. The figure shows the comparison with the previous frameworks.
<img src='imgs/teaser.png'>


## Installation

Some of the codes are built upon DenseCap: Fully Convolutional Localization Networks for Dense Captioning [[website]](https://cs.stanford.edu/people/karpathy/densecap/). We appreciate them for their great work.
Our code is implemented in [Torch](http://torch.ch/), and depends on the following packages: [torch/torch7](https://github.com/torch/torch7), [torch/nn](https://github.com/torch/nn), [torch/nngraph](https://github.com/torch/nngraph), [torch/image](https://github.com/torch/image), [lua-cjson](https://luarocks.org/modules/luarocks/lua-cjson), [qassemoquab/stnbhwd](https://github.com/qassemoquab/stnbhwd), [jcjohnson/torch-rnn](https://github.com/jcjohnson/torch-rnn). You'll also need to install
[torch/cutorch](https://github.com/torch/cutorch) and [torch/cunn](https://github.com/torch/cunn);

After installing torch, you can install / update these dependencies by running the following:

```bash
luarocks install torch
luarocks install nn
luarocks install image
luarocks install lua-cjson
luarocks install https://raw.githubusercontent.com/qassemoquab/stnbhwd/master/stnbhwd-scm-1.rockspec
luarocks install https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/torch-rnn-scm-1.rockspec
luarocks install cutorch
luarocks install cunn
luarocks install cudnn
```

## Pre-trained model
You can download a pretrained Relational Captioning model from this link: [Pre-trained model](https://drive.google.com/file/d/19t6Ogcl_ZlW9G6sPLBiWXfepWlX7MXg3/view?usp=sharing):

Download the model and place it in the root folder.

This is not the exact model that was used in the paper, but with different hyperparameters. it achieve a recall of 36.25 on the test set which is better than the reall of 34.27 that we report in the paper.


## Evalation
To evaluate a model on our Relational Captioning Dataset, you will following the following steps:

1. Download the raw images from [the Visual Genome website](https://visualgenome.org/api/v0/api_home.html)
2. Download our relational captioning label from the following link: [Dataset](https://drive.google.com/file/d/1cCN36poslxe7cCMkLnhYK0a-Y3vO4Rfn/view?usp=sharing).
3. Use the script `preprocess.py` to generate a single HDF5 file containing the entire dataset.
4. Use the script `evaluate_model.lua` to evaluate a trained model on the validation or test data.


## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{densecap,
  title={Dense Relational Captioning: Triple-Stream Networks for Relationship-Based Captioning},
  author={Kim, Dong-Jin and Choi, Jinsoo and Oh, Tae-Hyun and Kweon, In So},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```
