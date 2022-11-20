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

This project can generate face sketch from photos using the GAN-based model.
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
* You need to process the file into the format of the /dataset/list_train_sample.txt directory (mat1 and mat2 represent the results of the segmentation process of photos and sketches, respectively）



* Train a model

  ```
  python train.py --dataset_train_list train_sample.txt --dataset_test_list test_sample.txt   --name eadt
  ```

* Test the model

  ```
  python test.py  --input_size 256  --checkpoint_dir /home/sd01/EADT/checkpoint/eadt.ckpt
  ```


### Preprocessing steps

If you need to use your own data, please align all faces by eyes and the face parsing is segmented by [face-parsing](https://github.com/jehovahxu/face-parsing.PyTorch)


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
