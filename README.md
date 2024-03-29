# Dense Relational Image Captioning

The code for our [CVPR 2019](https://cvpr2019.thecvf.com/) paper along with our journal extention paper ([arXiv](https://arxiv.org/abs/2010.03855)),

**[Dense Relational Captioning: Triple-Stream Networks for Relationship-Based Captioning](https://sites.google.com/view/relcap)**.

Done by Dong-Jin Kim, Jinsoo Choi, Tae-Hyun Oh, and In So Kweon.

Link: **[arXiv](https://arxiv.org/pdf/1903.05942.pdf)** , **[arXiv (Journal Extension)](https://arxiv.org/abs/2010.03855)** , **[Dataset](https://drive.google.com/file/d/1cCN36poslxe7cCMkLnhYK0a-Y3vO4Rfn/view?usp=sharing)**, **[Pre-trained model](https://drive.google.com/file/d/19t6Ogcl_ZlW9G6sPLBiWXfepWlX7MXg3/view?usp=sharing)**.


<img src='imgs/teaser.png'>
We introduce “relational captioning,” a novel image captioning task which aims to generate multiple captions with respect to relational information between objects in an image. The figure shows the comparison with the previous frameworks.

## Updates
(28/08/2019)
- Our code is updated from evaluation-only to trainable version.
- Codes for backpropagation part are added to several functions.

(06/09/2019)
- Fixed the bug of UnionSlicer code.
- Added eval_utils_mAP.lua.

(24/12/2020)
- Added the code for running our model on new images.

(26/01/2021)
- Updated the model in the journal extension version (MTTSNet+REM).

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
You can download a pretrained Relational Captioning model from this link: Pre-trained models for either [CVPR19 version](https://drive.google.com/file/d/19t6Ogcl_ZlW9G6sPLBiWXfepWlX7MXg3/view?usp=sharing) or [Journal version](https://drive.google.com/file/d/1iIGJ78krcxmh9NApzt4QzKj9efA-NnEB/view?usp=sharing).

Download the model and place it in `./`.

This is not the exact model that was used in the paper, but with different hyperparameters. it achieve a recall of 36.2 on the test set which is better than the reall of 34.27 that we report in the paper.


## Running on new images

To run the model on new images, use the script `run_model.lua`. To run the pretrained model on an image,
use the following command:

```bash
th run_model.lua -input_image /path/to/my/image/file.jpg
```

By default this will run in GPU mode; to run in CPU only mode, simply add the flag `-gpu -1`.

If you have an entire directory of images on which you want to run the model, use the `-input_dir` flag instead:

```bash
th run_model.lua -input_dir /path/to/my/image/folder
```

This run the model on all files in the folder `/path/to/my/image/folder/` whose filename does not start with `.`.


## Evaluation
To evaluate a model on our Relational Captioning Dataset, please follow the following steps:

1. Download the raw images from Visual Genome dataset version 1.2 [website](https://visualgenome.org/api/v0/api_home.html). Place the images in `./data/visual-genome/VG_100K`.
2. Download our relational captioning label from the following link: [Dataset](https://drive.google.com/file/d/1cCN36poslxe7cCMkLnhYK0a-Y3vO4Rfn/view?usp=sharing). Place the json file at `./data/visual-genome/1.2/`.
3. Use the script `preprocess.py` to generate a single HDF5 file containing the entire dataset.
4. Run `script/setup_eval.sh` to download and unpack METEOR jarfile.
5. Use the script `evaluate_model.lua` to evaluate a trained model on the validation or test data either with CVPR 2019 version (MTTSNet):
```bash
th evaluate_model.lua -checkpoint checkpoint_VGlongv3_tLSTM_MTL2_1e6.t7
```
or with recent journal extension version (MTTSNet+REM):
```bash
th evaluate_model.lua -checkpoint checkpoint_VGlongv3_REM_tLSTM_MTL2_512_FC+nonlinear_1e6.t7
```
6. If you want to measure the mAP metric, change the line9 from `imRecall` to `mAP` and run `evaluate_model.lua`.

## Training
To train a model on our Relational Captioning Dataset, you can simply follow these steps:

1. Run `script/download_models.sh` to download VGG16 model.
2. Run `train.lua` to train a relational captioner. As default, the option -REM is set to be 1 which is for the journal version model (MTTSNet+REM). If you want to train a CVPR19 version model (MTTSNet), set the option -REM to be 0:
```bash
th train.lua -REM 0
```
The Recall and METEOR scores for the provided model for are as follows:
|Model|Recall|METEOR|
|:-|:-:|:-:|
|Direct Union [1] |17.32|11.02|
|Neural Motifs [2] |29.90|15.34|
|**MTTSNet (Ours)**|**34.27**|**18.73**|
|**MTTSNet (Ours) + REM**|**45.96**|**18.44**|

**References:**

[1] Johnson, J., Karpathy, A., & Fei-Fei, L. (2016). Densecap: Fully convolutional localization networks for dense captioning. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4565-4574).

[2] Zellers, R., Yatskar, M., Thomson, S., & Choi, Y. (2018). Neural motifs: Scene graph parsing with global context. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 5831-5840).


## Citation
If you find our work useful in your research, please consider citing our CVPR2019 paper or our TPAMI version paper:
```
@inproceedings{kim2019dense,
  title={Dense relational captioning: Triple-stream networks for relationship-based captioning},
  author={Kim, Dong-Jin and Choi, Jinsoo and Oh, Tae-Hyun and Kweon, In So},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={6271--6280},
  year={2019}
}

@article{kim2021dense,
  title={Dense relational image captioning via multi-task triple-stream networks},
  author={Kim, Dong-Jin and Oh, Tae-Hyun and Choi, Jinsoo and Kweon, In So},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021},
  publisher={IEEE}
}
```
