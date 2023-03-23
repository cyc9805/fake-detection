# Fake_detection
__'병원 영수증 날짜 위조 여부 판별'__ 프로젝트에 사용된 코드입니다.

# 사전 훈련된 모델 생성하기
[zi2zi](https://github.com/EuphoriaYan/zi2zi-pytorch) 모델의 인코더만 활용하므로 파인튜닝을 더 쉽게 하기 위해 별도의 모델을 만든 뒤 해당 모델에 pre-trained된 parameter를 불러왔습니다. 별도의 모델을 생성하기 위해서는 아래와 같은 과정을 진행해야 합니다.
1. [Baidu](https://pan.baidu.com/s/1wRiDg_vOY7EMWZHQLRJcpw)에서 90000_net_G.pth 파일을 다운 받은 뒤 experiment/checkpoint 폴더에 넣습니다.
2. 다음 코드를 실행합니다.
  ```shell
  cd run_scripts
  bash create_pt_model.sh
  ```
# 오려붙이기 판별을 위한 모델 훈련하기
아래 코드를 실행합니다. 모델의 성능에 대한 그래프는 results 폴더에 저장이 됩니다.
  ```shell
  cd run_scripts
  bash run_instance.sh
  ```
  
# 직접타이핑 판별 방법에 대한 성능 평가하기
아래 코드를 실행합니다. 위조된 영수증과 레퍼런스 패치와의 코사인 유사도 히트맵은 sim_maps 폴더에 저장이 됩니다.
```shell
  cd run_scripts
  bash chk_accuracy.sh
  ```
#### __개인정보 문제로 데이터는 업로드 하지 않았습니다__
