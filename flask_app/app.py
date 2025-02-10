from flask import Flask, render_template, Blueprint, request, jsonify
from predict_final import load_and_predict
from transformers import BertTokenizer


app = Flask(__name__, template_folder="templates", static_folder="static")

# Blueprint 정의
predict_api = Blueprint('predict', __name__)

@predict_api.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get('payload')  # 클라이언트로부터 페이로드 받기
        
        if not data:
            app.logger.error("No payload provided")
            return jsonify({"error": "No payload provided"}), 400
        
        app.logger.info(f"Received payload: {data}")
        
        model_path = "C:/Projects/web/flask_app/best_model_bert.keras"  # 실제 경로로 수정 필요
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # 디버깅 정보 추가
        app.logger.info(f"Loading model from {model_path}")
        
        # 모델을 사용해 예측
        predicted_label, predicted_probability, class_probabilities = load_and_predict(model_path, data, tokenizer)
        app.logger.info("Prediction completed successfully")

        return jsonify({
            "payload": data,
            "predicted_label": predicted_label,
            "predicted_probability": predicted_probability,
            "class_probabilities": class_probabilities
        })
    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500
    
# Blueprint 등록
app.register_blueprint(predict_api)

@app.route('/')              # main 경로
def home():
    return render_template('main.html')

@app.route('/payloadInput')  # /payloadInput 경로 추가
def payload_input():
    return render_template('payloadInput.html')  # payloadInput.html 렌더링

@app.route('/payloadOutput')  # /payloadOutput 경로 추가
def payload_output():
    return render_template('payloadOutput.html')  # payloadOutput.html 렌더링

@app.route('/payloadResult')  # /payloadResult 경로 추가
def payload_result():
    return render_template('payloadResult.html')  # payloadResult.html 렌더링

@app.route('/register')  # /register 경로 추가
def register():
    return render_template('register.html')  # register.html 렌더링

@app.route('/forgotPassword')  # /register 경로 추가
def forgot_password():
    return render_template('forgotPassword.html')  # register.html 렌더링

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9091, debug=True)
