# [Adversarial Learning For Semi-Supervised Semantic Segmentation](https://openreview.net/references/pdf?id=HkTgsG-CW "Open Review")
## ICLR Reproducibility Challenge 2018

### Baseline Model
| Name| Details | mIoU |
| --- | --- | --- |
|101| - No Normalization <br>  - No gradient for batch norm <br> - Drop last batch if not complete <br> - Volatile = false for eval <br> - Poly Decay every 10 iterations | 35.91 |
| 102 | Exactly like 101, except <br> - no polynomial decay <br> - **3 fixed bilinear upsampling layers**| 68.84|
|103|Exactly like 102, except<br>with polynomial decay(every 10 iter))|68.88(e14)|
|104| Exactly like 103, except <br> with poly decay every step| **69.78**(e15)|
