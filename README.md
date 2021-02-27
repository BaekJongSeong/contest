# tensorflow-yolov4
This is a project where I participated in "the online hackathon for Dynamic Objects Detection" hosted by the Ministry of Science_ICT, NIA and hosted by Aimmo.

If the hackathon is divided into 3 part("the idea part, the technology part, and the commercialization part"), this part is a repository corresponding to the technology part and explains tensorflow-yolov4.

Download yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT

이 프로젝트는 제가 과학기술정보통신부 및 NIA한국정보화진흥원이 주관하고, Aimmo가 주최하는 동적객체인지 온라인 해커톤에 참가했던 프로젝트입니다.

해커톤을 3부분(아이디어 파트와 기술파트, 사업화파트)로 나누어보면, 이 부분은 기술파트에 해당하는 repository이며 yolov4에 대해서 설명하고 있습니다.

------
# 프로젝트 전체적 개요입니다
제공받은 데이터(png 10329장 + json 파일 10329개) 이용
---
|데이터 전처리|학습데이터셋 생성|학습|모델 생성 및 예측|
|Aimmo측에서 제공해준 전체 데이터 중에서 정확한 데이터셋 확보를 위해 10329개의 json 파일 중, 365개의 empty annotation으로 이루어진 json 파일 제거. 
최종 9964개의 data로 추림.
Labels 중에서 목적차량을 차량으로 통합(최종 분류 class 개수 3개)
mAP 정확도 향상을 위해 epoch를 class 개수 *20000으로 총 60000번 지정. (최종 학습완료된 epoch는 12000번)
데이터가 방대하기 때문에 Image argumentation을 적용하지 않음. (똑같은 사진을 회전과 좌우 반전을 부여하여 data 개수 늘림으로써 모델 성능이 정확)
|a|a|a|

---

```bash
## yolov4
python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4 

```
데이터 전처리부터 model train / deteck 과정 전주기 과정은 darknet_yolov4_final_contest.py와 tensorflow_yolov4_final.py를 참고하시기 
