# 이 코드는 커스텀 Keras 레이어를 정의하는 코드입니다.
# 주로 텐서플로우 모델에서 BERT 모델을 임베딩 레이어로 사용하려는 경우 사용됩니다.

import tensorflow as tf
from transformers import TFBertModel

# @tf.keras.utils.register_keras_serializable()는 Keras가 이 레이어를 직렬화할 수 있게 해주는 데코레이터입니다.
# 이 레이어는 Keras 모델에 포함되어 훈련 중이나 추론 중에 BERT 임베딩을 처리할 수 있습니다.
@tf.keras.utils.register_keras_serializable()
class BertEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        # Keras Layer의 기본 생성자를 호출하여 부모 클래스 초기화
        super(BertEmbeddingLayer, self).__init__(**kwargs)
        
        # BERT 모델을 로드합니다. 'bert-base-uncased'는 사전 학습된 BERT 모델입니다.
        self.bert = TFBertModel.from_pretrained('bert-base-uncased')

    def call(self, inputs):
        # call() 메서드는 모델이 데이터를 처리할 때 호출됩니다.
        # inputs는 입력으로 들어오는 두 가지 텐서: input_ids와 attention_mask입니다.
        input_ids, attention_mask = inputs
        
        # BERT 모델을 통해 입력 텍스트를 처리합니다. 이때 'input_ids'와 'attention_mask'를 사용하여
        # BERT 모델의 출력을 얻고, 마지막 숨겨진 상태를 반환합니다.
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 마지막 숨겨진 상태 반환 (입력 시퀀스에 대한 BERT의 임베딩)
        return outputs.last_hidden_state

    def get_config(self):
        # 레이어의 구성(config)을 반환합니다. Keras 모델을 저장하거나 로드할 때 필요한 정보입니다.
        config = super(BertEmbeddingLayer, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        # config를 사용하여 레이어를 복원할 때 사용됩니다. 모델을 로드할 때 호출됩니다.
        return cls(**config)
