# Adversarial Learning For Semi-Supervised Semantic Segmentation

### This is a submission for ICLR 2018 Reproducibility Challenge.

## TODO
- [ ] Set up the segmentation network (ResNet-101, pre-trained on ImageNet)
  - [ ] Remove the last classification layer
  - [ ] Modify the stride of the last two classification layers from 2 to 1
  - [ ] Apply dilated convolution to conv4 (stride 2) and conv5 (stride 4)
  - [ ] Add 3x3 dilated (6) convolution layer after the last layer
  - [ ] Add final classifier layer to output C channel feature map
  - [ ] Apply 8x (bilinear) upsampling layer on the last layer with softmax
