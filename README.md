[ KISA AI 보안관제 전문인력 양성과정 (팀 프로젝트) ]
1. 주어진 데이터셋을 분석하여, AI를 이용해 자동으로 네트워크 공격유형을 분류하는 모델을 설계하고, 이를 웹서비스로 구현할 수 있도록 고도화 함.
2. model 로드 부분이 불안정하여 'model 저장부터 로드'하는 과정까지 재구현 필요한 상태로 보류함.
3. 참고사항
- predict_final -> model_train.py에서 학습된 모델을 이용하여 예측만 수행하는 코드 (Flask에 사용)
- model_train.py -> BERT 임베딩 모델, CNN+RNN 하이브리드 신경망을 이용하여 모델 학습하는 코드 (참고용, Flask 적용 X)
- best_model_bert.keras -> model_train.py에서 훈련된 모델을 추출한 파일 (모델)
