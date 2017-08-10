# Optimization as a Model for Few-Shot Learning
This repo provides a Pytorch implementation for the [Optimization as a Model for Few-Shot Learning](https://openreview.net/pdf?id=rJY0-Kcll) paper.

## Installation of pytorch
The experiments needs installing [Pytorch](http://pytorch.org/)

## Data 
For the miniImageNet you need to download the ImageNet dataset and execute the script utils.create_miniImagenet.py changing the lines:
```
pathImageNet = '<path_to_downloaded_ImageNet>/ILSVRC2012_img_train'
pathminiImageNet = '<path_to_save_MiniImageNet>/miniImagenet/'
```
And also change the main file option.py line or pass it by command line arguments:
```
parser.add_argument('--dataroot', type=str, default='<path_to_save_MiniImageNet>/miniImagenet/',help='path to dataset')
```

## Installation

    $ pip install -r requirements.txt
    $ python main.py 
    

## Acknowledgements
Special thanks to @sachinravi14 for their Torch implementation. I intend to replicate their code using Pytorch. More details at https://github.com/twitter/meta-learning-lstm

## Cite
```
@inproceedings{Sachin2017,
  title={Optimization as a model for few-shot learning},
  author={Ravi, Sachin and Larochelle, Hugo},
  booktitle={In International Conference on Learning Representations (ICLR)},
  year={2017}
}
```


## Authors

* Albert Berenguel (@aberenguel) [Webpage](https://scholar.google.es/citations?user=HJx2fRsAAAAJ&hl=en)
