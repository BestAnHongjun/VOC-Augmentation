"""
This file is under Apache License 2.0, see more details at https://www.apache.org/licenses/LICENSE-2.0
Author: Coder.AN, contact me at an.hongjun@foxmail.com
Github: https://github.com/BestAnHongjun/VOC-Augmentation
"""

import os
import matplotlib.pyplot as plt

# import VocAug
from VocAug.voc_aug import voc_aug
from VocAug.transform.voc2vdict import voc2vdict
from VocAug.utils.viz_bbox import viz_vdict

if __name__ == "__main__":
    voc2vdict_transformer = voc2vdict()
    augmentation_transformer = voc_aug()

    # prepare the xml-file-path and the image-file-path
    filename = "000007"
    file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    xml_file_path = os.path.join(file_dir, "Annotations", "{}.xml".format(filename))
    image_file_path = os.path.join(file_dir, "JPEGImages", "{}.jpg".format(filename))

    # Firstly convert the VOC format xml&image path to VOC-dict(vdict), then augment it.
    src_vdict = voc2vdict_transformer(xml_file_path, image_file_path)
    image_aug_vdict = augmentation_transformer(src_vdict)

    # take out the image
    image_src = src_vdict.get("image")
    image_src_with_bbox = viz_vdict(src_vdict)

    image_aug = image_aug_vdict.get("image")
    image_aug_with_bbox = viz_vdict(image_aug_vdict)

    # visualization
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.title("src")
    plt.imshow(image_src)
    plt.subplot(2, 2, 3)
    plt.title("src_bbox")
    plt.imshow(image_src_with_bbox)
    plt.subplot(2, 2, 2)
    plt.title("aug")
    plt.imshow(image_aug)
    plt.subplot(2, 2, 4)
    plt.title("aug_bbox")
    plt.imshow(image_aug_with_bbox)
    plt.show()
