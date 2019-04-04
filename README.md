Cascaded Partial Decoder for Fast and Accurate Salient Object Detection (CVPR2019)
====

Requirements: 
----
python2.7, pytorch 0.4.0

Usage
-----
Modify the pathes of backbone network, datasets, and run test_CPD.py

Pre-trained model
-----
VGG16     backbone: [google drive](https://drive.google.com/open?id=1ddopz30_sNPOb0MvTCoNwZwL-oQDMGIW)

ResNet50  backbone: [google drive](https://drive.google.com/open?id=188sybU9VU5rW2BH2Yzhko4w-G5sPp6yG)

Saliency Maps
-----
VGG16     backbone: [google drive](https://drive.google.com/open?id=1LcCTcKGEsZjO8WUgbGpiiZ4atQrK1u_O)

ResNet50  backbone: [google drive](https://drive.google.com/open?id=16pLY2qYZ1KIzPRwR7zFUseEDJiwhdHOg)

Performance
-----
Maximum F-measure

|Model|ECSSD|HKU-IS|DUT-OMRON|DUTS-TEST|PASCAL-S|
|:----|:----|:----|:----|:----|:----|
|PiCANet|0.931|0.921|0.794|0.851|0.862|
|CPD|0.936|0.924|0.794|0.864|0.866|
|PiCANet-R|0.935|0.919|0.803|0.860|0.863|
|CPD-R|0.939|0.925|0.797|0.865|0.864|



Shadow Detection
-----
results: [google drive](https://drive.google.com/open?id=1R__w0FXpMhUMnIuoxPaX6cFzwAypX13U)
