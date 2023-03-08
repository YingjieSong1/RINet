# RINet
This project provides the code and results for **'RINet: Relative Importance-Aware Network for Fixation Prediction'**, IEEE TMM 2023. [Paper link](https://ieeexplore.ieee.org/document/10054110).

## Abstract
Fixation prediction aims to simulate human visual selection mechanism and estimate the visual saliency degree of regions in a scene. In semantically rich scenes, there are generally multiple salient regions. This condition requires a fixation prediction model to understand the relative importance relationship of multiple salient regions, that is, to identify which region is more important. In practice, existing fixation prediction models implicitly explore the relative importance relationship in the end-to-end training process while they do not work well. In this article, we propose a novel Relative Importance-aware Network (RINet) to explicitly explore the modeling of relative importance in fixation prediction. RINet perceives multi-scale local and global relative importance through the Hierarchical Relative Importance Enhancement (HRIE) module. Within a single scale subspace, on the one hand, HRIE module regards the similarity matrix as the local relative importance map to weight the input feature. On the other hand, HRIE module integrates a set of local relative importance maps into one map, defined as the global relative importance map, to grasp global relative importance. Moreover, we propose a Complexity-Relevant Focal (CRF) loss for network training. As such, we can progressively emphasize learning difficult samples for better handling the complicated scenarios, further improving the performance. The ablation studies confirm the contributions of key components of our RINet, and extensive experiments on five datasets demonstrate our RINet is superior to 28 relevant state-of-the-art models.

## Model Architecture
<p align="center">
<img src="https://github.com/Mango321321/RINet/blob/main/image/network.png" width=100% height=100%>
</p>

## Requirements
Create  an anaconda environment:

```commandline
$ conda env create -f environment.yml
```

Activate the environment:

```commandline
$ source activate torch36
$ <run_python_command> # see the examples below
```

## Results Download
Prediction results on **SALITON-Test**,**MIT300**, **PASCAL-S**, **TORONTO**, and **DUT-OMRON** can be downloaded from:

Baidu Disk: <https://pan.baidu.com/s/1HfZNfNAsKqzJRAbX4WU7eA>  (password:`s216`)

## Citation
        @ARTICLE{10054110,
                author={Song, Yingjie and Liu, Zhi and Li, Gongyang and Zeng, Dan and Zhang, Tianhong and Xu, Lihua and Wang, Jijun},
                journal={IEEE Transactions on Multimedia}, 
                title={RINet: Relative Importance-Aware Network for Fixation Prediction}, 
                year={2023},
                volume={},
                number={},
                pages={1-15},
                doi={10.1109/TMM.2023.3249481}}
