# tensorflow-yolov4(2021.01.20~2021.02.20)

This is a project where I participated in "the online hackathon for Dynamic Objects Detection" hosted by the Ministry of Science_ICT, NIA and hosted by Aimmo.

If the hackathon is divided into 3 part("the idea part, the technology part, and the commercialization part"), this part is a repository corresponding to the technology part and explains tensorflow-yolov4.

이 프로젝트는 제가 과학기술정보통신부 및 NIA한국정보화진흥원이 주관하고, Aimmo가 주최하는 동적객체인지 온라인 해커톤에 참가했던 프로젝트입니다.

해커톤을 3부분(아이디어 파트와 기술파트, 사업화파트)로 나누어보면, 이 부분은 기술파트에 해당하는 repository이며 yolov4에 대해서 설명하고 있습니다.

------
# 프로젝트 전체 개요

자율주행차량을 통해 제공되는 10329장의 image 데이터(공단에서 지정해준 사고다발지역)와 그에 해당하는 10329개의 json파일을 이용

object detection을 진행하기 위한 인공지능 모델 학습 후 test image 데이터들에 대한 모델 정확도를 테스트하는 형식

실습환경
1. Python 3.6.9 + GPU Tesla V100 + Ram 25.5GB 사양 사용을 위해 colab 사용

git clone하기 전에 다운받아야할 파일 + 명령어
### 경로는 지정 파일 위치로 사용자별로 바꿔야함
```bash
#git clone
%cd /content/
!git clone https://github.com/hunglc007/tensorflow-yolov4-tflite.git

# 다운받을 yolov4.weights 파일
Download yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT
https://drive.google.com/file/d/1Y5JP2bn2I-Woqwsi-qhSs3WmGtKlwmov/view

#이미 준비된 weights 파일이 있다면 아래 명령어를 통해서 변경
!python save_model.py --weights /content/drive/My\ Drive/Colab\ Notebooks/darknet/bin/darknet/backup_01/sia_best.weights --output ./mAP/yolov_hack-416 --input_size 416 --model yolov4

#학습 명령
!python train.py --weights ./data/sia_best.weights

#detect명령
!python detect.py --weights ./mAP/yolov_hack-416 --size 416 --model yolov4 --image /content/ab.png

#파일 여러개 한번에 detect.py 돌릴떄
def letsgo(file):
  !python detect.py --weights ./mAP/yolov4-416 --size 416 --model yolov4 --image $file

for file in files:
  file=path+file
  letsgo(file)

#files는 전체 png 파일 위치
```
------
전체 과정
### 1. 데이터 전처리
### 2. Config.py 파일 구성 및 hyperparameter수정
### 3. 모델 학습 train.py
### 4. detect.py를 통한 예측
---
+ 데이터 전처리
  + Aimmo측에서 제공해준 전체 데이터 중에서 정확한 데이터셋 확보를 위해 10329개의 json 파일 중, 365개의 empty annotation으로 이루어진 json 파일 제거. 
  + 최종 9964개의 data로 추림. (총 분류하려는 class 개수 3개로 조정: 자동차 사람 이륜차)
  + mAP 정확도 향상을 위해 epoch를 class 개수 *20000으로 총 60000번 지정. (최종 학습완료된 epoch는 12000번)
  + 데이터가 방대하기 때문에 Image argumentation을 적용하지 않음. (똑같은 사진을 회전과 좌우 반전을 부여하여 data 개수 늘림으로써 모델 성능이 정확)|
  + 각 bounding box마다의 좌표들을 image size 1920x1080에 대한 0~1 사이의 범위로 scaling 하는 과정 거침
  ### + Yolo(You Only Look Once)를 사용하기 위해서 json 파일을 txt 파일로 변환.

+ tensorflow 파일 구성 핵심
  + Core
    + backbone.py
    + common.py
    + config.py
    + dataset.py
    + utils.py
    + yolov4.py
  + detect.py
  + save_model.py
  + train.py
---
+ Core
  + Config.py
    + __C.YOLO.CLASSES: class = 3이므로 클래스 이름이 들어있는 파일 위치 지정
    + __C.TRAIN.ANNOT_PATH & __C.TEST.ANNOT_PATH : image data의 train/valid 데이터 셋 및 names 파일 위치 정보
    + __C.TRAIN.BATCH_SIZE: 각 epoch 별 batch size 지정
	  + batch, subdivision크기와 학습 이미지 크기 조정, hyperparameter 조정, learning rate와 epoch, [convolutional neural network] 및 [yolo network]의 filter와 activation 조정, max pooling size 조정
	
  + dataset.py
    + 파일이름에 띄어쓰기가 있는 경우 처리.
    + Coco annotation으로 되어있으면 이미지 파일 이름 + left top, right bottom 좌표 + class_id가 들어오는데 이때 띄어쓰기를 기준으로 들어옴.
    + 그래서 dataset 오류 생길 수 있으므로 image 파일 이름 띄어쓰기를 처리

  + yolov4.py
    + Subdivision = 64 (1 epoch 당 batch size)
    + Width & height = 416
    + momentum=0.949 (optimazation으로 과거 이동 방식 기억, 가중치 계속 부여)
    + channels=3	(1 epoch 당 3개의 channels)
    + learning_rate=0.001(학습 속도가 너무 느려도, 빨라도 안됨)
    + CNN activation=mish (relu activation보다 부드럽게 올라가는 gradient)
    + 마지막 부분의 CNN activation = linear
    + YOLO layer은 이미지의 feature들을 뽑고 난 후에 실질적인 prediction을 하는 layer. 
    + Mask: 총 9개의 anchor이 정의되어 있는데, 그 중 mask에 적혀있는 tag에 해당하는 anchor들만 사용.(ex) anchors = 12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401)

+ 생성된 모델을 통한 detect.py
