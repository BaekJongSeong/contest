# tensorflow-yolov4

Download yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT

```bash
## yolov4
python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4 

```
데이터 전처리부터 model train / deteck 과정 전주기 과정은 darknet_yolov4_final_contest.py와 tensorflow_yolov4_final.py를 참고하시기 
