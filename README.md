# VocAug
A transformer which can randomly augment VOC format dataset online.

### The highlight is, 
1) it augments both image and b-box!!!
2) it only use cv2 & numpy, means it could be used simply without any other awful packages!!!
3) it is an online transformer!!!

### It contains methods of:
1) Random HSV augmentation
2) Random Cropping augmentation
3) Random Flipping augmentation
4) Random Noise augmentation
5) Random rotation or translation augmentation

All the methods can adjust abundant arguments in the constructed function of class VocAug.voc_aug.

### Here are some visualized examples:
![eg1](examples/000007.png)
![eg2](examples/000009.png)