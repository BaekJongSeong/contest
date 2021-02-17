# -*- coding: utf-8 -*-
"""tensorflow_yolov4_final.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1etVRiyVBmTmNdsFKf5Rns15b8gFAprkC
"""

!git config --global user.email "[jsbaek01@ajou.ac.kr]"
!git config --global user.name "[BaekJongSeong]"

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/

# Commented out IPython magic to ensure Python compatibility.
# %cd

!ls

!git add tensorflow-yolov4-tflite

!git init

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/
!git clone https://github.com/ultralytics/yolov5.git  # clone repo

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/yolov5/
!pip install -r requirements.txt

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd /
from glob import glob   #images를 trainimages로만 바꾸면 됨
img_list = glob('/content/drive/MyDrive/Colab Notebooks/yolov5/yolov5/dataset/export/images/*.jpg')
print(len(img_list))

from sklearn.model_selection import train_test_split
train_img_list, val_img_list = train_test_split(img_list,test_size=0.2,random_state=2000)
print(len(train_img_list),len(val_img_list))

with open('/content/drive/MyDrive/Colab Notebooks/yolov5/yolov5/dataset/train.txt','w') as f:
  f.write('\n'.join(train_img_list) + '\n')

with open('/content/drive/MyDrive/Colab Notebooks/yolov5/yolov5/dataset/val.txt','w') as f:
  f.write('\n'.join(val_img_list) + '\n')

import yaml

with open('/content/drive/MyDrive/Colab Notebooks/yolov5/yolov5/dataset/dynamic.yaml','r') as f:      #dynamic.yaml가 문제인가?
  data = yaml.load(f)

print(data)
data['train'] = '/content/drive/MyDrive/Colab Notebooks/yolov5/yolov5/dataset/train.txt'
data['val'] = '/content/drive/MyDrive/Colab Notebooks/yolov5/yolov5/dataset/val.txt'

with open('/content/drive/MyDrive/Colab Notebooks/yolov5/yolov5/dataset/dynamic.yaml','w') as f:
  yaml.dump(data,f)

print(data)

# Commented out IPython magic to ensure Python compatibility.
# %cd '/content/drive/MyDrive/Colab Notebooks/yolov5/yolov5/'
!python train.py --img 416 --batch 16 --epoch 6000 --data ./dataset/dynamic.yaml --cfg ./models/yolov5x.yaml --weights yolov5x.pt --name dynamic_yolov5x_results

"""**자, 자꾸 버전 업을 부탁했어. yolov5에 대한 버전업을 부탁했으므로 그에 응하지말고**

그냥 yolov5를 또 하나 다운받아 (대신 로컬에 말고 여기 런타임에만)

그리고 나서 저장 위치만 싹다 로컬로 바꿔버려

그리고 jpg로 바꿔버리자

**최종 jpg와 txt파일의 합작으로**
"""

# Commented out IPython magic to ensure Python compatibility.
# %cd '/content/yolov5/'
!python train.py --data /content/drive/MyDrive/Colab\ Notebooks/yolov5/yolov5/dataset/dynamic.yaml --cfg /content/drive/MyDrive/Colab\ Notebooks/yolov5/yolov5/models/yolov5x.yaml --weights /content/drive/MyDrive/Colab\ Notebooks/yolov5/yolov5/runs/train/dynamic_yolov5x_results4/weights/last.pt --hyp /content/drive/MyDrive/Colab\ Notebooks/yolov5/yolov5/runs/train/dynamic_yolov5x_results4/hyp.yaml --name /content/drive/MyDrive/Colab\ Notebooks/yolov5/yolov5/runs/train/dynamic_yolov5x_results --img 416 --batch 16 --epoch 6000

"""# **yolo_v4 tensorflow 버전**"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/
!git clone https://github.com/hunglc007/tensorflow-yolov4-tflite.git

"""모델 변경하기 전에 폴더 내의 config.py에서 수정할 것 수정해야지.

yolov4에서 수정할 부분은 딱 2개. sia 파일과 sia_pic == dynamic 파일. 이중에서도 일단 

sia 파일의 data, names, cfg, weights. 이중에서도 names파일과 data파일이지

"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/tensorflow-yolov4-tflite/
#%cd /content/drive/MyDrive/tensorflow-yolov4-tflite/
#!python save_model.py --weights ./data/sia_last.weights --output ./mAP/yolov4_yester-416 --input_size 416 --model yolov4
!python save_model.py --weights /content/drive/MyDrive/tensorflow-yolov4-tflite/data/sia_best.weights --output ./mAP/yolov4-416 --input_size 416 --model yolov4

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/
!cp -r tensorflow-yolov4-tflite/ /content/drive/MyDrive/Colab\ Notebooks/yolov4/

import tensorflow as tf

tf.__version__

"""**train 시켜보자**

Traning your own model
## Prepare your dataset
## If you want to train from scratch:
In config.py set FISRT_STAGE_EPOCHS=0 
## Run script:
python train.py

## Transfer learning: 
python train.py --weights ./data/yolov4.weights

**먼저 train_.txt 파일 변경하기
"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /
from glob import glob
img_list = glob('/content/drive/MyDrive/tensorflow-yolov4-tflite/dataset/export/images/*.png')
print(len(img_list))

from sklearn.model_selection import train_test_split
train_img_list, val_img_list = train_test_split(img_list,test_size=0.2,random_state=2000)
print(len(train_img_list),len(val_img_list))

with open('/content/drive/MyDrive/tensorflow-yolov4-tflite/dataset/train.txt','w') as f:
  f.write('\n'.join(train_img_list) + '\n')

with open('/content/drive/MyDrive/tensorflow-yolov4-tflite/dataset/val.txt','w') as f:
  f.write('\n'.join(val_img_list) + '\n')

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/Colab Notebooks/yolov5/
!unzip 01.zip

import os

path = '/content/drive/MyDrive/Colab Notebooks/yolov5/'
files = os.listdir(path)
count=0
name=[]
name_txt=[]
what=[]
for file in files :
  if not 'json' in file:
    continue
  a=str(file)
  b=a.split('.')
  name.append(b[0])
name.sort()
print(len(name))

path = '/content/drive/MyDrive/tensorflow-yolov4-tflite/dataset/export/labels/'
files = os.listdir(path)
for file in files :
  if not 'txt' in file:
    continue
  a=str(file)
  b=a.split('.')
  name_txt.append(b[0])
name_txt.sort()
print(len(name_txt))
for i in range(10329):
  count1=0                #flag가 중요
  for j in range(9964):
    if name[i] == name_txt[j]:
      count1+=1
      break
  if count1==0:
    what.append(name[i])

print(len(what))
what

#what 배열을 사용하기.
path = '/content/drive/MyDrive/Colab Notebooks/yolov5/'
files = os.listdir(path)
for file in files :
  if 'json' in file:
    for a in what:
      if str(file) == (a+'.png.json'):
        path2 = path+a+'.png.json'
        os.remove(path2)

import os
import json  
from PIL import Image
#==============================================================================
def convert(size, box): 
  dw = 1./size[0] 
  dh = 1./size[1] 
  x = (box[0] + box[1])/2.0 
  y = (box[2] + box[3])/2.0 
  w = box[1] - box[0] 
  h = box[3] - box[2] 
  x =x*dw 
  w =w*dw 
  y =y*dh 
  h =h*dh
  return (x,y,w,h) 
#==============================================================================
path = '/content/drive/MyDrive/Colab Notebooks/yolov5/'
files = os.listdir(path)
dic={
    '일반차량' : 0,
    '목적차량(특장차)' : 0,
    '이륜차' : 1,
    '보행자' : 2
}
image_id='ab'
all=[];point=list()
count = 0;class_id=0;x=0;y=0;width=0;height=0       #count will be 1200
#==============================================================================
for file in files :
  if not 'json' in file:
    continue
  with open(path+file,'r') as input:     
    data=input.read()
  json_data = json.loads(data,strict=False)
  image_id=json_data['filename']
  height=json_data['metadata']['height']
  width=json_data['metadata']['width']
  #=============================================================================
  for properties in json_data['annotations']:          #annotation의 value들이 순차적으로 돌아감 (중괄호 3개)
      class_id=dic[properties['label']]                #중괄호 하나 안에 5개의 key들이 있는데 
      point=properties['points'].copy()        #2차원 list
      xmin=10000;xmax=0;ymin=10000;ymax=0
      for x in point:
        if (xmin > x[0]): xmin = x[0]
        if xmax < x[0]: xmax = x[0]
      for y in point:
        if (ymin > y[1]): ymin = y[1]
        if ymax < y[1]: ymax = y[1]
      b = (xmin, ymin, xmax, ymax)
      #bb = convert((width,height), b)
      line=[]
      line.append(class_id);line.append(b[0]);line.append(b[1]);line.append(b[2]);line.append(b[3])
      all.append(line)
  for i in image_id:
        if i == '.':
          image_id=image_id.split('.')[0]
          break
  f = open("/content/drive/MyDrive/tensorflow-yolov4-tflite/dataset/label/"+image_id+".txt", 'w')
  for i in range(len(all)):
    f.write(' '.join(map(str,all[i])))
    f.write('\n')
  f.close()
  all.clear()
  count+=1
print(count)

"""**나는 png, txt 파일을 나눠놨지**

## **아래 코드는 기존 darknet yolov4의 txt파일을 읽어서 coco annotation txt로 바꾸는 코드**
"""

import os
path = '/content/drive/MyDrive/tensorflow-yolov4-tflite/dataset/label/'
path1 = '/content/drive/MyDrive/tensorflow-yolov4-tflite/dataset/train.txt'
path2 = '/content/drive/MyDrive/tensorflow-yolov4-tflite/dataset/val.txt'
all1=[];all2=[]
files = os.listdir(path)

def convert(x,y,w,h): 
  box=[0]*4
  dw = 1./1920
  dh = 1./1080
  x =x/dw 
  w =w/dw 
  y =y/dh 
  h =h/dh
  box[0]=round((2*x-w)/2)
  box[1]=round(x+w/2)
  box[2]=round((2*y-h)/2)
  box[3]=round(y+h/2)
  return box[0],box[1],box[2],box[3]

with open(path1,'r') as input:        #with 블럭이 끝나면 자동으로 닫아줌
  data=input.readline()
  while data:
    data=(data.strip('\n'))
    all1.append(data)       #읽어올때 \n까지 읽어오자나
    data = input.readline()
    if len(data)<=1:break

with open(path2,'r') as input:        #with 블럭이 끝나면 자동으로 닫아줌
  data2=input.readline()
  while data2:
    data2=(data2.strip('\n'))
    all2.append(data2)
    data2 = input.readline()
    if len(data2)<=1:break

print(len(all1))
print(len(all2))

f=open(path1,'w')
for i in range(len(all1)):
  for file in files:
    name = ((all1[i].split('/'))[8].split('.'))[0]
    if not name == (str(file).split('.'))[0]:
      continue
    line=[]
    f.write(all1[i])
    f.write(' ')
    with open(path+file,'r') as input:     #파일들 하나씩 나와서
      data=input.readline()       #'\n' 같이 들어오겠지?
      while data:
        data=(data.strip('\n'))
        data=list(map(float,data.split()))                   #한줄씩 리스트로 읽기
        line.append(data)
        data = input.readline()
        if len(data)<=1:break
    
    for j in range(len(line)):
      line2=[]
      #line[j][1],line[j][2],line[j][3],line[j][4] = convert(line[j][1],line[j][2],line[j][3],line[j][4])
      line2.append(int (line[j][1]));line2.append(int (line[j][2]));line2.append(int (line[j][3]));line2.append(int (line[j][4]));line2.append(int (line[j][0]))
      f.write(','.join(map(str,line2)))
      f.write(' ')
    f.write('\n')
f.close()

f=open(path2,'w') 
for i in range(len(all2)):
  for file in files:
    name = ((all2[i].split('/'))[8].split('.'))[0]
    if not name == (str(file).split('.'))[0]:
      continue
    line=[]
    f.write(all2[i])
    f.write(' ')
    with open(path+file,'r') as input:     #파일들 하나씩 나와서
      data=input.readline()
      while data:
        data=(data.strip('\n'))
        data=list(map(float,data.split()))                   #한줄씩 리스트로 읽기
        line.append(data)
        data = input.readline()
        if len(data)<=1:break

    for j in range(len(line)):
      line2=[]
      #line[j][1],line[j][2],line[j][3],line[j][4] = convert(line[j][1],line[j][2],line[j][3],line[j][4])
      line2.append(int (line[j][1]));line2.append(int (line[j][2]));line2.append(int (line[j][3]));line2.append(int (line[j][4]));line2.append(int (line[j][0]))
      f.write(','.join(map(str,line2)))
      f.write(' ')
    f.write('\n')
f.close()

"""**보관용**"""

# Commented out IPython magic to ensure Python compatibility.
#따른거 손댈 필요 없이 config만 손대면 된다는 건가??? 아닐꺼 같은뎨~~
#생각해보자. names파일, config 파일, data파일, weights 파일 + train_txt 파일 이렇게가 수정해야할 것들
# %cd '/content/drive/MyDrive/tensorflow-yolov4-tflite/'
!python train.py --weights ./data/sia_best.weights

# Commented out IPython magic to ensure Python compatibility.
# %cd '/content/drive/MyDrive/tensorflow-yolov4-tflite/'
!python train.py --weights ./data/sia_best.weights

"""**오 tensorflow 내꺼 모델로도 충분히 car 62%찍히고, 원래 darknet으로 돌리면 car63% 찍히네**"""

# Commented out IPython magic to ensure Python compatibility.
import os
# %cd '/content/tensorflow-yolov4-tflite/'
#!python detect.py --weights ./mAP/yolov4_yester-416 --size 416 --model yolov4 --image /content/abc.png
#!/bin/bash
#path= '/content/drive/MyDrive/Colab Notebooks/darknet/bin/darknet/dynamic/01.학습데이터자료/'#'/content/drive/MyDrive/result/result_source/'
#path2= '/content/drive/MyDrive/Colab\ Notebooks/darknet/bin/darknet/dynamic/01.학습데이터자료/'
path= '/content/drive/MyDrive/result/result_source/'
files = os.listdir(path)
files.sort()
#filess=[]
#for file in files:
#  if '.png' in file:
#    filess.append(file)

#print(len(filess))

files=files[:1000].copy()
def letsgo(file):
  !python detect.py --weights ./mAP/yolov4-416 --size 416 --model yolov4 --image $file

for file in files:
  file=path+file
  letsgo(file)







"""# 새 섹션

# 새 섹션

# 새 섹션
"""























"""# 새 섹션"""











"""**이 밑은 잠시 darknet으로도 정확도 테스트 해본거**"""

# Commented out IPython magic to ensure Python compatibility.
#darknet 권한 변경 및 실행 테스트
# %cd /content/drive/My\ Drive/Colab\ Notebooks/darknet/bin/darknet
!chmod +x ./darknet
!./darknet detector

# Commented out IPython magic to ensure Python compatibility.
#!./darknet detector train sia/sia.DATA sia/sia.cfg backup_05/sia_last.weights -map -dont_show
# %cd /content/drive/MyDrive/Colab Notebooks/darknet/bin/darknet/
!./darknet detector test sia/sia.DATA sia/sia.cfg /content/drive/MyDrive/tensorflow-yolov4-tflite/data/sia_best.weights /content/ab.png

!./darknet detector test sia/sia.DATA sia/sia.cfg /content/drive/MyDrive/tensorflow-yolov4-tflite/data/sia_best.weights /content/ab.png -dont_show -ext_output /content/drive/MyDrive/tensorflow-yolov4-tflite/data/result.txt

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/result/
!unzip result_source.zip