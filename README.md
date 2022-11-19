# EADT

> Face sketch synthesis, image-to-image translation, generative adversarial network, edge-aware domain transformation, edge-preserving filter.

We provide `PyTorch` implementations for our TIFS2022 paper [`EADT`](https://ieeexplore.ieee.org/abstract/document/9845477): 

```latex
@article{zhang2022edge,
  title={Edge Aware Domain Transformation for Face Sketch Synthesis},
  author={Zhang, Congyu and Liu, Decheng and Peng, Chunlei and Wang, Nannan and Gao, Xinbo},
  journal={IEEE Transactions on Information Forensics and Security},
  volume={17},
  pages={2761--2770},
  year={2022},
  publisher={IEEE}
}
```

This project can generate face sketch  from photos using the GAN-based model.
[`Paper@IEEE`](https://ieeexplore.ieee.org/abstract/document/9845477)   [`Code@Github`](https://github.com/shenhaiyoualn/EADT)  

## Framework
<div align=center><img width="1076" height="376" src=imgs/fig1.png/></div>



## Prerequisites

- Linux 
- Python 3.7
- Pytorch-lightning 0.7.5
- NVIDIA GPU + CUDA + CuDNN

## Getting Started

### Installation

* Clone this repo: 

  ```
  git clone https://github.com/shenhaiyoualn/EADT
  cd EADT
  ```

* Install all the dependencies by:

  ```
  pip install -r requirements.txt
  ```

### train/test

* Download CUFS and CUFSF dataset and put it in the dataset directory
* Download parsing [model](https://drive.google.com/file/d/1VNEoJEZLLdvX-cP0xohyWv_xYqNtprjU/view?usp=share_link) and move it to /model/parsing/cp/
* You need to process the file into the format of the /dataset/list_train_sample.txt directory



* Train a model

  ```
  # python train.py --dataset_train_list train_sample.txt --dataset_test_list test_sample.txt   --name eadt
  ```

* Test the model

  ```
  python test.py  --input_size 256  --checkpoint_dir /home/sd01/EADT/checkpoint/eadt.ckpt
  ```


### Apply a pre-trained model

- A face $photo \mapsto sketch$ model pre-trained on [CUHK/CUFS]( https://drive.google.com/drive/folders/1A1LABGQKUAN9tYunb8tHjsASLxSL313D?usp=sharing)
- The [pre-trained model](https://drive.google.com/drive/folders/1lrb-K4_xuMGLYka-G3hAJ1Y3iZefuBcD?usp=sharing) need to be save at `./checkpoint`
- Then you can test the model

### Preprocessing steps

Face photos (and paired drawings) need to be aligned and have facial parsing. And facial parsing after alignment are needed in our code in training. 

In our work,facial parsing is segmented by method in [1]

* First,  we need to align, resize and crop face photos (and corresponding drawings) to 256x256 
* Then,we use code in https://github.com/zllrunning/face-parsing.PyTorch to detect facial parsing for face photos and drawings. 

> [1] Yu, Changqian, et al. "**Bisenet**: Bilateral segmentation network for real-time semantic segmentation." *Proceedings of the European conference on computer vision (ECCV)*. 2018.

## Demo

<video width="480" height="270" controls>

<source src="demo.mp4" type="video/mp4">

</video>

Our portrait drawing demos: (a) QR code of the applet of WeChat, (b) QR code of the Web API, (c)-(e) layouts of the applet of WeChat, and (f) picture of the drawing robot. Readers can try our demos by scanning the QR codes.

![](imgs/demo.jpg)

## Citation

 If you use this code for your research, please cite our paper. 

> Zhang, C., Liu, D., Peng, C., Wang, N., & Gao, X. (2022). Edge Aware Domain Transformation for Face Sketch Synthesis. IEEE Transactions on Information Forensics and Security, 17, 2761-2770. (Accepted)

**bibtex:**

```latex
@article{zhang2022edge,
  title={Edge Aware Domain Transformation for Face Sketch Synthesis},
  author={Zhang, Congyu and Liu, Decheng and Peng, Chunlei and Wang, Nannan and Gao, Xinbo},
  journal={IEEE Transactions on Information Forensics and Security},
  volume={17},
  pages={2761--2770},
  year={2022},
  publisher={IEEE}
}
```

## Acknowledgments

Our code is inspired by [GENRE](https://github.com/fei-hdu/genre) and [SPADE/GauGAN](https://github.com/NVlabs/SPADE).
