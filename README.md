## Introduction
[SnowVision](https://github.com/rudylyh/SnowVision) is a project of exploring the application of deep learning and computer vision techniques in traditional archeological heritage fragment recognition. It is part of the World Engraved project ([http://worldengraved.org](http://worldengraved.org/index)).

The aim of SnowVision is to automate the matching process between fragmentary stamped pottery sherds from the archaeological record and their appropriate complete paddle design from among the hundreds of registered design reconstructions in the corpus. Example of sherds and designs are shown below.

<p align="center">
<img src="data/eg_img.png" width="80%"/>

Technically, our goal is to find the ground truth design of the query sherd in the design database and the corresponding location.

<p align="center">
<img src="data/match_goal.png"/>
</p>

## How to use
#### Requirements:
- caffe==1.0.0
- pcl==1.8.0
- opencv==3.4.1
- skimage==0.10.1

#### Input:
#### Output:
#### Before running:
#### Usage example:

## Citation
```
@inproceedings{lu2018curve,
  title={Curve-structure segmentation from depth maps: A CNN-based approach and its application to exploring cultural heritage objects},
  author={Lu, Yuhang and Zhou, Jun and Wang, Jing and Chen, Jun and Smith, Karen and Wilder, Colin and Wang, Song},
  booktitle={Thirty-Second AAAI Conference on Artificial Intelligence},
  year={2018}
}
@incollection{zhou2019framework,
  title={A Framework for Design Identification on Heritage Objects},
  author={Zhou, Jun and Lu, Yuhang and Smith, Karen and Wilder, Colin and Wang, Song and Sagona, Paul and Torkian, Ben},
  booktitle={Proceedings of the Practice and Experience in Advanced Research Computing on Rise of the Machines (learning)},
  pages={1--8},
  year={2019}
}
```
