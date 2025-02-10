import tensorflow as tf
from tensorflow.keras import layers, models
from transformers import TFBertModel, BertTokenizer
import math

# 예측 클래스 라벨을 매핑한 딕셔너리 (숫자 -> 라벨)
LABEL_MAP = {
    0: "Scanning",
    1: "Brute_Force",
    2: "System_Cmd_Execution",
    3: "SQL_Injection",
    4: "Cross_Site_Scripting",
    5: "normal"
}

# BERT 임베딩 레이어 정의
class BertEmbeddingLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(BertEmbeddingLayer, self).__init__(**kwargs)
        self.bert = TFBertModel.from_pretrained('bert-base-uncased')

    def call(self, inputs):
        input_ids, attention_mask = inputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state  # (batch_size, sequence_length, embedding_dim)

    def compute_output_shape(self, input_shape):
        batch_size, seq_length = input_shape[0][0], input_shape[0][1]
        return (batch_size, seq_length, 768)  # Hidden size = 768

    def get_config(self):
        config = super(BertEmbeddingLayer, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls()


# 입력 데이터 변환 함수
def convert_to_bert_input(texts, tokenizer, max_length=128):
    encodings = tokenizer(
        texts,
        truncation=True,  # 텍스트가 길 경우 잘라냄
        padding='max_length',  # max_length로 패딩 추가
        max_length=max_length,  # max_length로 길이 조정
        return_tensors='tf'  # TensorFlow 형식으로 반환
    )
    return encodings['input_ids'], encodings['attention_mask']


# 저장된 모델 로드 및 예측 함수
def load_and_predict(model_path, text, tokenizer):
    # 모델 로드 시 커스텀 레이어 등록
    
    try:
        #model_path = "C:/Projects/web/flask_app/best_model_bert.keras"
        print(f"load and predict ::: Trying to load model from: {model_path}")
        model = tf.keras.models.load_model(model_path, custom_objects={'BertEmbeddingLayer': BertEmbeddingLayer})
        print("load and predict ::: Model loaded successfully.")
    except Exception as e:
        print(f"load and predict ::: Failed to load model. Error: {e}")
        raise e
    
    try:
        # 입력 데이터 변환
        input_ids, attention_mask = convert_to_bert_input([text], tokenizer)

        # 예측 수행
        predictions = model.predict([input_ids, attention_mask])
        
        # 예측된 클래스 (가장 높은 확률을 가진 클래스)
        predicted_class = tf.argmax(predictions, axis=1).numpy()[0]

        # 예측된 클래스의 라벨
        predicted_label = LABEL_MAP[predicted_class]

        # 예측 확률
        predicted_probability = tf.reduce_max(predictions, axis=1).numpy()[0]
        predicted_probability = predicted_probability * 100

        # 각 클래스별 예측 확률 (백분율로 변환)
        class_probabilities = predictions[0] * 100  # 확률을 100으로 곱해 백분율로 변환

        # 확률을 소수점 2자리로 버림 처리
        class_probabilities = [math.floor(prob * 100) / 100.0 for prob in class_probabilities]

        return predicted_label, predicted_probability, class_probabilities
    except Exception as e:
        print(f"load and predict ::: Error during prediction: {e}")
        raise e


# 실행 예시

# if __name__ == "__main__":
#     # 토크나이저 및 모델 경로 설정
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     model_path = "C:\Projects\web\flask_app\best_model_bert.keras"  # 학습된 모델 경로 (colab으로 경로이기 때문에 Flask로 만들때 수정해야합니다.)
#     sample_text = "POST HTTP/1.1 /admin/login?userid=admin&password=123456" # 여기에 웹 입력값이 들어가도록 처리하면 될듯 합니다.

#     # 예측 실행
#     predicted_label, predicted_probability, class_probabilities = load_and_predict(model_path, sample_text, tokenizer)

#     print(f"예측 라벨: {predicted_label}, 예측 확률: {predicted_probability:.2f}%")

#     # 각 클래스별 예측 확률을 백분율로 출력
#     print("각 클래스별 예측 확률 (백분율):")
#     for i, prob in enumerate(class_probabilities):
#         print(f"{LABEL_MAP[i]}: {prob}%")

        
    '''
    predict_final (예측)
    payload_input -> predict 파일의 sample_text 변수에 저장
    
    flask -> 메인 파이썬 파일 app.py
    
    if __name__ -> 메인 파이썬 파일이 지금 열려있는 파일(predict.py) 동작-> preidct.py 파일만 단독으로 실행했을 경우에만 동작함
    app.py 돌릴 경우 위 구문 동작 X
    
    -> python predict.py 명령어로 실행하면 if __name__ 하위 구문이 동작하지만
    -> python app.py? 명령어로 실행하면 동작 X
    
    '''
