# *FGD Implemented on YOLOv5 (PyTorch & Jittor)*

---
## Introduction
This code provides implementation of *Focal and Global Distillation for Detectors [FGD, CVPR-2022](https://arxiv.org/abs/2111.11837)* on YOLOv5 (PyTorch & Jittor).

---
## Environment
Please refer to [requirements.txt](requirements.txt).

---
## Dataset
This code uses the publicly available remote sensing dataset [NWPU VHR-10](https://ieeexplore.ieee.org/document/7560644).<br>
The process of the dataset can be seen in [data/transform_nwpu.py](data/transform_nwpu.py).<br>
You can also download this dataset in [here](https://www.kaggle.com/datasets/kefanzhan/nwpudataset), where the dataset has already been split.

---
## Train & Test
The Pytorch and Jittor script of *Train & Test* can be seen in [here](YOLOv5Pytorch/train.py) and [here](YOLOv5Jittor/train.py), respectively. <br>
First, you should set the mode as "train", and train a powerful teacher model (like using yolov5s.yaml). <br>
Then, train a lightweight student model (like using yolov5n.yaml). <br>
Finially, to achieve FGD, set the mode as "KD" and provides teacher model accordingly.<br>
In each period, the script will test the model when finishing training.<br>
More specific training details can be found in each script. <br>
It is worth mentioning that we do not use pretrained model in each period.

--- 
## Visualization
To visualize the Focal Distillation in FGD, we have presented some visual results in [FGD-visualize](FGD-visualize). <br>
And we also provide the script of visualization in [here](YOLOv5Pytorch/visualize.py).

--- 
## Problems
We have concluded some Q&A in [QuestionAndAnswer.md](QuestionAndAnswer.md), hope they might help.

---
## Results
In this section, we select mAP50 (mean Average Precision at 50% IoU) as metric indicator. <br>
More visual and digital outcomes (e.g. loss curves and logs) can be seen in each result file,<br>
for instance, [YOLOv5Pytorch/TeacherTraining/runs/train/exp/results.png](YOLOv5Pytorch/TeacherTraining/runs/train/exp/results.png) and [YOLOv5Pytorch/TeacherTraining/output.log](YOLOv5Pytorch/TeacherTraining/output.log). 

|         |                PyTorch                |                Jittor                |
|:-------:|:-------------------------------------:|:------------------------------------:|
| Teacher | [90.3](YOLOv5Pytorch/TeacherTraining) | [59.5](YOLOv5Jittor/TeacherTraining) |
| Student | [86.3](YOLOv5Pytorch/StudentTraining) | [53.8](YOLOv5Jittor/StudentTraining) |
|Student+FGD|   [87.9](YOLOv5Pytorch/StudentFGD)    |   [58.6](YOLOv5Jittor/StudentFGD)    |

Unfortunately, the YOLOv5 (Jittor) performs not well, <br>
and we have not found the problem yet after controlling the model's architecture and hyperparameters in corresponding with the PyTorch version.

---
## More information
We have also incorporated FGD with YOLOv8 (PyTorch) in [here](https://github.com/KefanZhan/YOLOv8-KD) ⭐! <br>
Despite FGD, that code also assembles other interesting knowledge distillation methods.

---
## Reference
1. https://github.com/yzd-v/FGD
2. https://github.com/ultralytics/yolov5
3. https://github.com/li-xl/yolo.jittor
4. https://github.com/Jittor/jittor

---
## Acknowledgements
This code is built on [YOLOv5 (PyTorch)](https://github.com/ultralytics/yolov5) and [YOLOv5 (Jittor)](https://github.com/li-xl/yolo.jittor). <br>
We thank the authors for sharing the codes.

---
## Contact
If you have any questions, please contact me by email (kefanzhan@smail.xtu.edu.cn).