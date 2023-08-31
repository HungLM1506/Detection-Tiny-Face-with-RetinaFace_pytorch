# RetinaFace in PyTorch

A [PyTorch](https://pytorch.org/) implementation of [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641). Model size only 1.7M, when Retinaface use mobilenet0.25 as backbone net. We also provide resnet50 as backbone net to get better result. The official code in Mxnet can be found [here](https://github.com/deepinsight/insightface/tree/master/RetinaFace).

## Data

1. Organise the dataset directory as follows:

```Shell
  ./data/widerface/
    train/
      images/
      label.txt
    val/
      images/
      wider_val.txt
```

2. We also provide the organized dataset we used as in the above directory structure
   Link: from [google cloud](https://drive.google.com/open?id=11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS) or [baidu cloud](https://pan.baidu.com/s/1jIp9t30oYivrAvrgUgIoLQ) Password: ruck

## Installation

```Shell
git clone https://github.com/HungLM1506/Detection-Tiny-Face-with-RetinaFace_pytorch.git
cd Pytorch_Retinaface
pip install -r requirement.txt
```

## Training

We provide restnet50 and mobilenet0.25 as backbone network to train model.
We trained Mobilenet0.25 on imagenet dataset and get 46.58% in top 1. If you do not wish to train the model, we also provide trained model. Pretrain model and trained model are put in [Here](https://drive.google.com/drive/folders/1odpdS9XtU4DMIp8S7HdNPyBMqVNroLkB?usp=drive_link) . The model could be put as follows:

```Shell
  ./weights/
      mobilenet0.25_Final.pth
      mobilenetV1X0.25_pretrain.tar
      Resnet50_Final.pth
```

## Run

```Shell
python detect.py --trained_model weights/Resnet50_Final.pth or Mobilenet0.25_Final --network resnet50 or mobile0.25
```

## Evaluation widerface val

1. Generate txt file

```Shell
python test_widerface.py --trained_model weight_file --network mobile0.25 or resnet50
```

2. Evaluate txt results. Demo come from [Here](https://github.com/wondervictor/WiderFace-Evaluation)

```Shell
cd ./widerface_evaluate
python setup.py build_ext --inplace
python evaluation.py
```

## References

- [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)
- [Retinaface (mxnet)](https://github.com/deepinsight/insightface/tree/master/RetinaFace)

```
@inproceedings{deng2019retinaface,
title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
booktitle={arxiv},
year={2019}
```
