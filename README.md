# [Adversarial Learning For Semi-Supervised Semantic Segmentation](https://openreview.net/references/pdf?id=HkTgsG-CW "Open Review")
## Introduction
This is a submission for ICLR 2018 Reproducibility Challenge. The central theme of the work by the authors is to incorporate adversarial training for semantic-segmentation task which enables the segmentation-network to learn in a semi-supervised fashion on top of the traditional supervised learning. The authors claim significant improvement in the performance (measured in terms of mean IoU) of segmentation network after the supervised-training is extended with adversarial and semi-supervised training.

## Scope
 My plan is to reproduce the improvement in the performance of the segmentation network (Resnet-101) by including adversarial and semi-supervised learning scheme over the baseline supervised training and document my experience along the way. The authors have used two datasets, PASCAL VOC 12 (extended version) and Cityscapes, to demonstrate  the benefits of their proposed training scheme. I will focus on PASCAL VOC 12 dataset for this work. Specifically, the target for this work is to reproduce the following table from the paper.

 | Method | &emsp; &emsp; &emsp; Data Amount <br> 1/2 &emsp; &emsp; &emsp; full |
 | --- | --- |
 | Baseline (Resnet-101) | 69.8 &emsp; &emsp; &emsp;73.6  |
 |Baseline + Adversarial Training| 72.6 &emsp; &emsp; &nbsp;  74.9|
 |Baseline + Adversarial Training + <br> Semi-supervised Learning|73.2 &emsp; &emsp; &nbsp;  NA|

## Results Reproduced
Following table summarizes the results I have been able to reproduce for the full dataset. For the full dataset, only the performance of the adversarial training on top of baseline can be evaluated.


| Method (Full Dataset) | Original | Challenge |
| --- | --- | --- |
| Baseline (Resnet-101) | 73.6  | 69.98 |
|Baseline + Adversarial Training|  74.9| 70.97 |
|Baseline + Adversarial Training + <br> Semi-supervised Learning|NA| NA|

Following table summarizes the results I was able to reproduce for the semi-supervised training where half of the training data is reserved for semi-supervised training with unlabeled data. 

| Method (1/2 Dataset) | Original | Challenge |
| --- | --- | --- |
| Baseline (Resnet-101) | 69.8  | 67.84 |
|Baseline + Adversarial Training|  72.6| 68.89 |
|Baseline + Adversarial Training + <br> Semi-supervised Learning|73.2| 69.05|


## Updates

* ***18th Dec 2017*** Finished Refactoring of the code, re-ran the experiments and achieved some improvement on the previous results by training the network longer.

* ***8th Dec 2017***: Semi-supervised Learning with 1/2 of training data treated as unlabeled degrades the performance compare to baseline (68.05 mIoU) and baseline + adversarial training (70.31 mIoU). It might be related to one of the comments of the reviewer that initial predictions by the discriminator might be noisy which renders semi-supervised training unstable during early epochs. The authors have made a comment that semi-supervised training is only applied after 5k iterations. I'll include the results with this addition soon.

* ***4th Dec 2017***: Started working on Semi-supervised training.

* ***2nd Dec 2017***: Adversarial Training based on base105 improves mIoU from 68.86 to 69.93.

* ***30th Nov 2017***: Managed to improve adversarial training performance. For base105, mIoU was improved from **68.86** to **69.33**.

* ***28th Nov 2017***: Started experiments with Imagenet-pretrained Resnet-101 segmentation network as the baseline. Best mIoU achieved is 65.97. So, moving forward to unsupervised training with the base104 (best baseline model) and base105 (baseline with best adversarial training results).

* ***27th Nov 2017***: Finally managed to stabilize the GAN training. I couldn't reproduce any significant improvement over the baseline Segmentation Network. In fact, the best performing segmentation network (base104 with mIoU 69.78) was worse off with the adversarial training (mIoU dropped to 68.07). I have documented the details of the experiments performed for adversarial training. As GAN training is considered to be very sensitive towards weight initialization, I feel this is the right time to incorporate ImageNet pretrained network in the training.

* ***20th Nov 2017***: Started working on adding adversarial learning for base-104 segmentation network.

* ***17th Nov 2017***: Baseline model (base-104) achieved  mean IoU of **69.78** on the full dataset. The model is still significantly away from the target mIoU of 73.6. Only significant component missing from the implementation is using Resnet-101 pre-trained on Imagenet (I am currently using MS-COCO pretrained Network as the baseline). Other minor additions (~~normalization of the input~~ (included in base-105), number of iterations to wait before lr decay, etc) will also be included.  

## Journey
### Baseline Model
| Name| Details | mIoU |
| --- | --- | --- |
|base-101| - No Normalization <br>  - No gradient for batch norm <br> - Drop last batch if not complete <br> - Volatile = false for eval <br> - Poly Decay every 10 iterations <br> - learnable upsampling with transposed convolution  | 35.91 |
| base102 | Exactly like base-101, except <br> - no polynomial decay <br> - fixed bilinear upsampling layers| 68.84|
|base103|Exactly like base-102, except<br> - with polynomial decay(every 10 iter))|68.88|
|**base104**| Exactly like base-103, except <br> -with poly decay (every iter)| **69.78**|
|base105| base-104, except <br> - with normalization of input to 0 mean and unit variance| 68.86|
| base110 | - ImageNet pretrained <br> - Normalization <br> - poly decay(eveyr iter) <br> same lr for all layers| 65.97 |
| base111 | - Imagenent pretrained <br> - Normalization <br> - poly decay (every iter) <br> - 10x lr for classification module | 65.67 |
### Adversarial Models
|Name | Details | miou|
| --- | --- | --- |
| adv101| - base105 as G <br> - Optim(D): SGD lr 0.0001, momentum=0.5,decay= 0.0001 | 68.96 |
| adv102| - base105 <br> - 0.25 label smoothing for real labels in D <br> - Optim(D) SGD lr 0.0001, momentum=0.5,decay= 0.0001| 67.14|
| adv103 | - base105 <br> - 0.25 label smoothing for real labels in D <br> - Optim(D) ADAM | 68.07 |
| adv104 | - base104 <br> - 0.25 label smoothing for real labels in D <br> - Optim(D) SGD lr 0.0001, momentum=0.5,decay= 0.0001 |63.37 |
| adv105 | base104 as G <br> - everything else like adv103 | Very poor (didn't finish training) |
| adv105-cuda| - base105 <br> - 0.25 label smoothing for real labels in D <br> - Optim(D) SGD lr 0.0001, momentum=0.5,decay= 0.0001 <br> - batch size 21| Very poor (didn't finish training)|
| adv106| - base104 <br> - optim(D) ADAM <br> - batch_size = 21|61.50 |
| adv201| - base 105 <br> - label smoothing 0.25 <br> - Adam| 69.33|
| **adv202**| - base105 <br> - label smoothing 0.1 <br> - d_optim Adam | **69.93** |
| adv203 | - base105 <br> - label smoothing 0.1 <br> - Adam d_lr = 0.0001 and g_lr =  0.00025  | 69.72|
| adv204 | - base105 <br> - label smoothing 0.1 <br> - Adam d_lr = 0.00001, g_lr = 0.00025| 69.28|
