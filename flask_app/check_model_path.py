# 이 코드는 지정된 경로에 모델 파일이 존재하는지 확인하는 코드입니다.
# 주로 모델 파일 경로가 올바른지 확인하거나, 모델이 로드될 수 있는지 점검할 때 사용됩니다.

import os  # os 모듈은 파일 및 디렉토리 작업을 처리할 때 사용됩니다.

# 모델 파일 경로를 지정합니다.
model_path = "D:/project/web/MyFlaskApp/best_model_bert.keras"

# 지정된 경로에 모델 파일이 존재하는지 확인합니다.
if os.path.exists(model_path):
    # 경로에 파일이 존재하면, 해당 파일이 발견되었음을 출력합니다.
    print(f"Model file found at: {model_path}")
else:
    # 경로에 파일이 없으면, 파일이 존재하지 않음을 출력합니다.
    print(f"Model file not found at: {model_path}")
