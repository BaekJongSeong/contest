# tensorflow-yolov4
This is a project where I participated in "the online hackathon for Dynamic Objects Detection" hosted by the Ministry of Science_ICT, NIA and hosted by Aimmo.

If the hackathon is divided into 3 part("the idea part, the technology part, and the commercialization part"), this part is a repository corresponding to the technology part and explains tensorflow-yolov4.

Download yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT

이 프로젝트는 제가 과학기술정보통신부 및 NIA한국정보화진흥원이 주관하고, Aimmo가 주최하는 동적객체인지 온라인 해커톤에 참가했던 프로젝트입니다.

해커톤을 3부분(아이디어 파트와 기술파트, 사업화파트)로 나누어보면, 이 부분은 기술파트에 해당하는 repository이며 yolov4에 대해서 설명하고 있습니다.

```bash
## yolov4
python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4 

```
데이터 전처리부터 model train / deteck 과정 전주기 과정은 darknet_yolov4_final_contest.py와 tensorflow_yolov4_final.py를 참고하시기 
