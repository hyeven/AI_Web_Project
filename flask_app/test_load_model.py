# 이 코드는 'best_model_bert.keras' 파일을 로드하는 코드입니다.
# 모델 로딩에 실패한 경우 오류 메시지를 출력합니다.

import tensorflow as tf  # TensorFlow 라이브러리를 임포트합니다.

# 모델 파일 경로를 지정합니다.
model_path = "best_model_bert.keras"

try:
    # 모델을 로드합니다. 모델 파일이 올바르게 존재하면 로드됩니다.
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")  # 모델이 정상적으로 로드되면 성공 메시지 출력
except Exception as e:
    # 모델 로딩 중 오류가 발생하면 예외를 처리하고 오류 메시지 출력
    print(f"Error loading model: {e}")
