import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras import layers, models, optimizers, callbacks
from transformers import BertTokenizer, TFBertModel

# 데이터셋 로드 및 분리
data = pd.read_excel("/content/dataset.xlsx")  # 데이터셋 로드
X = data.drop(columns=["label_action"])  # target 열을 제외한 데이터 (피처)
y = data["label_action"]

# Train / Temporary 분리 (70% / 30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Temporary를 Validation / Test로 분리 (15% / 15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# 데이터 크기 출력
print("전체 데이터 크기:", X.shape[0])
print("훈련 데이터 크기:", X_train.shape[0])
print("검증 데이터 크기:", X_val.shape[0])
print("테스트 데이터 크기:", X_test.shape[0])

# BERT tokenizer와 모델 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# 커스텀 레이어 정의
class BertEmbeddingLayer(layers.Layer):
    def __init__(self, bert_model, **kwargs):
        super(BertEmbeddingLayer, self).__init__(**kwargs)
        self.bert = bert_model

    def call(self, inputs):
        input_ids, attention_mask = inputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size)

    def get_config(self):
        config = super().get_config()
        config.update({
            "bert_model": self.bert.name,  # BERT 모델의 이름을 저장
        })
        return config

# 텍스트 데이터 BERT 임베딩 변환 함수
def convert_to_bert_input(texts, tokenizer, max_length=128):
    """
    텍스트를 BERT 입력 형식으로 변환합니다.
    """
    encodings = tokenizer(
        texts, truncation=True, padding=True, max_length=max_length, return_tensors='np'
    )
    return encodings['input_ids'], encodings['attention_mask']

# X_train, X_val, X_test에서 텍스트 데이터만 추출하여 리스트로 변환
train_texts = X_train["payload"].tolist()  # 텍스트 컬럼 이름을 실제 데이터셋에 맞게 변경
val_texts = X_val["payload"].tolist()  # 텍스트 컬럼 이름을 실제 데이터셋에 맞게 변경
test_texts = X_test["payload"].tolist()  # 텍스트 컬럼 이름을 실제 데이터셋에 맞게 변경

# BERT 임베딩 추출
train_input_ids, train_attention_mask = convert_to_bert_input(train_texts, tokenizer)
val_input_ids, val_attention_mask = convert_to_bert_input(val_texts, tokenizer)
test_input_ids, test_attention_mask = convert_to_bert_input(test_texts, tokenizer)

# BERT 모델을 입력으로 하는 CNN + GRU 모델 정의
input_ids = layers.Input(shape=(128,), dtype=tf.int32, name='input_ids')
attention_mask = layers.Input(shape=(128,), dtype=tf.int32, name='attention_mask')

# 커스텀 BERT 레이어 사용
bert_embedding = BertEmbeddingLayer(bert_model)([input_ids, attention_mask])

# CNN + GRU 부분
x = layers.Conv1D(64, kernel_size=3, activation='relu')(bert_embedding)
x = layers.MaxPooling1D(pool_size=2)(x)
x = layers.GRU(128)(x)

# Dense Layer
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(len(y.unique()), activation='softmax')(x)

# 모델 정의
hybrid_model = models.Model(inputs=[input_ids, attention_mask], outputs=output)

# 모델 컴파일
hybrid_model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

# 체크포인트와 조기 종료 설정
checkpoint_path = "best_model_bert.keras"
model_checkpoint = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                             monitor='val_loss',
                                             save_best_only=True,
                                             verbose=1)

early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 모델 학습
history = hybrid_model.fit(
    [train_input_ids, train_attention_mask], y_train,
    epochs=10,  # 최대 10 에포크
    batch_size=32,
    validation_data=([val_input_ids, val_attention_mask], y_val),
    callbacks=[early_stop, model_checkpoint]
)

# 모델 평가
hybrid_eval = hybrid_model.evaluate([test_input_ids, test_attention_mask], y_test)
y_pred = np.argmax(hybrid_model.predict([test_input_ids, test_attention_mask]), axis=-1)
hybrid_acc = accuracy_score(y_test, y_pred)
classification_report_hybrid = classification_report(y_test, y_pred)

print(f"Hybrid Model Accuracy: {hybrid_acc}")
print("Classification Report:\n", classification_report_hybrid)

# 예측 함수
def predict_new_data(raw_text):
    """
    원본 텍스트(raw_text)를 받아 전처리, BERT 임베딩 후 예측 수행.
    """
    # 텍스트 전처리
    processed_text = raw_text.lower()

    # BERT 임베딩 수행
    input_ids, attention_mask = convert_to_bert_input([processed_text], tokenizer)

    # 예측
    hybrid_pred = np.argmax(hybrid_model.predict([input_ids, attention_mask]), axis=-1)
    hybrid_label = hybrid_pred[0]

    return {"Hybrid Prediction": hybrid_label}

# 테스트 데이터 예측
sample_text = "POST HTTP/1.1 /admin/login?userid=admin&password=123456"
prediction = predict_new_data(sample_text)
print("새로운 데이터 예측 결과:", prediction)
