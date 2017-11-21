# [Adversarial Learning For Semi-Supervised Semantic Segmentation](https://openreview.net/references/pdf?id=HkTgsG-CW "Open Review")
## Introduction
This is a submission (under development) for ICLR 2018 Reproducibility Challenge. The central theme of the work by the authors is to incorporate adversarial training for semantic-segmentation task which enables the segmentation-network to learn in a semi-supervised fashion on top of the traditional supervised learning. The authors claim significant improvement in the performance (measured in terms of mean IoU) of segmentation network after the supervised-training is extended with adversarial and semi-supervised training.

## Scope
 My plan is to reproduce the improvement in the performance of the segmentation network (Resnet-101) by including adversarial and semi-supervised learning scheme over the baseline supervised training and document my experience along the way. The authors have used two datasets, PASCAL VOC 12 (extended version) and Cityscapes, to demonstrate  the benefits of their proposed training scheme. I will focus on PASCAL VOC 12 dataset for this work. Specifically, the target for this work is to reproduce the following table from the paper.

 | Method | Data Amount <br> 1/8 &emsp; &emsp; &emsp; 1/4 &emsp; &emsp; &emsp; 1/2 &emsp; &emsp; &emsp; full |
 | --- | --- |
 | Baseline (Resnet-101) | 66.0 &emsp; &emsp;  68.3 &emsp; &emsp;  69.8 &emsp; &emsp; &emsp;73.6  |
 |Baseline + Adversarial Training|67.6 &emsp; &emsp; 71.0 &emsp; &emsp; &nbsp;     72.6 &emsp; &emsp; &nbsp;  74.9|
 |Baseline + Adversarial Training + <br> Semi-supervised Learning||


### Baseline Model
| Name| Details | mIoU |
| --- | --- | --- |
|101| - No Normalization <br>  - No gradient for batch norm <br> - Drop last batch if not complete <br> - Volatile = false for eval <br> - Poly Decay every 10 iterations | 35.91 |
| 102 | Exactly like 101, except <br> - no polynomial decay <br> - **3 fixed bilinear upsampling layers**| 68.84|
|103|Exactly like 102, except<br>with polynomial decay(every 10 iter))|68.88(e14)|
|104| Exactly like 103, except <br> with poly decay every step| **69.78**(e15)|
