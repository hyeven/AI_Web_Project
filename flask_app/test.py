import typing
typing.__getattr__ = lambda name: None
import tensorflow as tf
from transformers import TFBertModel

# 커스텀 레이어 정의
class BertEmbeddingLayer(tf.keras.layers.Layer): 
    def __init__(self, **kwargs):
        super(BertEmbeddingLayer, self).__init__(**kwargs)
        self.bert = TFBertModel.from_pretrained('bert-base-uncased')

    def call(self, inputs):
        input_ids, attention_mask = inputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

# SavedModel 디렉토리 경로
model_path = "C:/Projects/web/flask_app/best_model_bert_savedmodel" 

# 모델 로드
model = tf.keras.models.load_model(model_path, custom_objects={'BertEmbeddingLayer': BertEmbeddingLayer})

# 모델 정보 출력
print(model)
model.summary()
