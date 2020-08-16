# cv-papers-reading
This is a schedule and brief info about papers I read.

## Conference on Computer Vision and Pattern Recognition

1. [Deep Residual Learning for Image Recognition](https://scholar.google.co.in/scholar?oi=bibs&cluster=9281510746729853742&btnI=1&hl=en) <img src="https://img.shields.io/badge/Completed-Read%20on%2016--AUG--2020-green">

**Brief:** As Deeper neural networks are becoming challenging to train at that time, Residual Learning pushes and breaks all the SOTA results. It proposes a novel residual learning function instead of learning unreferenced functions. It involves a reformulation in stacking the layers for deeper architectures. In 2015 the best performing model was GoogleNet, which uses Inception modules and stacking them that worked well. As we continue to increase the depth of the models, they are not performing well. Yes, its the problem in 2015. For example, an 18-layer plain network giving better performance than a 34-layer plain network. As we increase the depth, the training error is increasing, and the test is also increasing. So we cannot take it as overfitting. The training error is less for overfitting, and the test error is more compared with less deep networks. ![Comparision for 20 and 56 layer networks](/resnet/comparision.png). You can observe that a 20-layer network is giving better results than a 56-layer network. The proposed residual learning and shortcut connections proved accuracy gains with an increase in depth.![Block of the residual net](/resnet/block.png). Shortcut connections are those which skip one or more layers. In this case, shortcut connections do Identity mapping, and their outputs are added to the outputs of stacked layers. They used batch normalization right after each convolution and before activation. They have not used dropout or max out. Absolutely the results are stunning and considered as a breakthrough in deep neural networks for visual recognition tasks.
![Results on Imagenet](/resnet/training.png) Even though plain networks suffered with an increase in depth, ResNets performed showing accuracy gains for increasing depths.They have implemented ResNets with 18,34,50,101,152 layers. ResNet152 giving the best accuracy among all networks With this they won 1st place in ILSVRC & COCO 2015 competitions:   Im-
ageNet detection, ImageNet localization, COCO detection,
and COCO segmentation.




2. [Going Deeper With Convolutions](https://scholar.google.co.in/scholar?oi=bibs&cluster=17799971764477278135&btnI=1&hl=en)<img src="https://img.shields.io/badge/Scheduled-Not%20Fixed-red">

3. [Fully Convolutional Networks for Semantic Segmentation](https://scholar.google.co.in/scholar?oi=bibs&cluster=16635967164511657165&btnI=1&hl=en)<img src="https://img.shields.io/badge/Scheduled-Not%20Fixed-red">

4. [You Only Look Once: Unified, Real-Time Object Detection](https://scholar.google.co.in/scholar?oi=bibs&cluster=6382612685700818764&btnI=1&hl=en)<img src="https://img.shields.io/badge/Scheduled-Not%20Fixed-red">

5. [Densely Connected Convolutional Networks](https://scholar.google.co.in/scholar?oi=bibs&cluster=4205512852566836101&btnI=1&hl=en)<img src="https://img.shields.io/badge/Scheduled-Not%20Fixed-red">

## International Conference on Computer Vision

1. [Fast R-CNN](https://scholar.google.co.in/scholar?oi=bibs&cluster=16324699838103945745&btnI=1&hl=en)<img src="https://img.shields.io/badge/Scheduled-Not%20Fixed-red">

2. [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://scholar.google.co.in/scholar?oi=bibs&cluster=6243061688889140249&btnI=1&hl=en)<img src="https://img.shields.io/badge/Scheduled-Not%20Fixed-red">

3. [Mask R-CNN](https://scholar.google.co.in/scholar?oi=bibs&cluster=11459229647356475672&btnI=1&hl=en)<img src="https://img.shields.io/badge/Scheduled-Not%20Fixed-red">

## European Conference on Computer Vision

1. [SSD: Single Shot MultiBox Detector](https://scholar.google.co.in/scholar?oi=bibs&cluster=15383553494348295625&btnI=1&hl=en)<img src="https://img.shields.io/badge/Scheduled-Not%20Fixed-red">

2. [Identity Mappings in Deep Residual Networks](https://scholar.google.co.in/scholar?oi=bibs&cluster=14035416619237709781&btnI=1&hl=en)<img src="https://img.shields.io/badge/Scheduled-Not%20Fixed-red">

3. [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://scholar.google.co.in/scholar?oi=bibs&cluster=5132755018694140583&btnI=1&hl=en)<img src="https://img.shields.io/badge/Scheduled-Not%20Fixed-red">

## IEEE Transactions on Pattern Analysis and Machine Intelligence

1. [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://scholar.google.co.in/scholar?oi=bibs&cluster=16436232259506318906&btnI=1&hl=en)<img src="https://img.shields.io/badge/Scheduled-Not%20Fixed-red">

2. [Fully Convolutional Networks for Semantic Segmentation](https://scholar.google.co.in/scholar?oi=bibs&cluster=16635967164511657165&btnI=1&hl=en)<img src="https://img.shields.io/badge/Scheduled-Not%20Fixed-red">

3. [Mask R-CNN](https://scholar.google.co.in/scholar?oi=bibs&cluster=11459229647356475672&btnI=1&hl=en)<img src="https://img.shields.io/badge/Scheduled-Not%20Fixed-red">

## IEEE Transactions on Image Processing


## 