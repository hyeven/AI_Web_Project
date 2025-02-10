predict_final -> model_train.py에서 학습된 모델을 이용하여 예측만 수행하는 코드 (Flask에 사용)
model_train.py -> BERT 임베딩 모델, CNN+RNN 하이브리드 신경망을 이용하여 모델 학습하는 코드 (참고용, Flask 적용 X)
best_model_bert.keras -> model_train.py에서 훈련된 모델을 추출한 파일 (쉽게, 모델 그 자체라고 생각하시면 됩니다. predict_final.py 와 같이 사용하면 됩니다.)