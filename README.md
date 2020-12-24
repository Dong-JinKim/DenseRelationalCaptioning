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
5. Use the script `evaluate_model.lua` to evaluate a trained model on the validation or test data.
6. If you want to measure the mAP metric, change the line9 from `imRecall` to `mAP` and run `evaluate_model.lua`.

## Training
To train a model on our Relational Captioning Dataset, you can simply follow these steps:

1. Run `script/download_models.sh` to download VGG16 model.
2. Run `train.lua` to train a relational captioner.


## Citation
If you find our work useful in your research, please consider citing our CVPR2019 paper or our journal extension paper:
```
@inproceedings{kim2019dense,
  title={Dense relational captioning: Triple-stream networks for relationship-based captioning},
  author={Kim, Dong-Jin and Choi, Jinsoo and Oh, Tae-Hyun and Kweon, In So},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={6271--6280},
  year={2019}
}
@article{kim2020dense,
  title={Dense Relational Image Captioning via Multi-task Triple-Stream Networks},
  author={Kim, Dong-Jin and Oh, Tae-Hyun and Choi, Jinsoo and Kweon, In So},
  journal={arXiv preprint arXiv:2010.03855},
  year={2020}
}
```
