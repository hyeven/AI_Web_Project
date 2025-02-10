# 이 코드는 'best_model_bert.keras' 파일을 로드하는 코드입니다.
# 모델을 로드할 때 BertEmbeddingLayer와 같은 커스텀 레이어를 정의하여 로드합니다.
# 모델 로딩에 실패한 경우 오류 메시지를 출력합니다.

from tensorflow.keras.models import load_model  # Keras 모델을 로드하는 함수를 임포트합니다.

# 모델 파일 경로를 지정합니다.
model_path = "D:/project/web/MyFlaskApp/best_model_bert.keras"

try:
    # 모델을 로드합니다. 
    # 'custom_objects'는 모델을 로드할 때 커스텀 레이어를 인식하기 위한 인자입니다.
    # compile=False는 모델을 컴파일하지 않고 로드하라는 옵션입니다. 
    # (주로 모델을 추론 용도로만 사용할 때 사용)
    model = load_model(
        model_path,
        custom_objects={"BertEmbeddingLayer": BertEmbeddingLayer},  # 커스텀 레이어 정의
        compile=False  # 컴파일을 하지 않고 모델만 로드
    )
    print("Model loaded successfully.")  # 모델이 정상적으로 로드되면 성공 메시지 출력
except Exception as e:
    # 모델 로딩 중 오류가 발생하면 예외를 처리하고 오류 메시지 출력
    print(f"Error loading model: {e}")
