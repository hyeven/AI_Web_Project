# 이 코드는 지정된 경로에 모델 파일이 존재하는지 확인하고, 그 결과(True/False)를 출력하는 코드입니다.
# 주로 파일의 존재 여부를 확인하고, 경로가 올바른지 확인할 때 사용됩니다.

import os  # os 모듈은 파일 경로 및 시스템 작업을 처리하는 데 사용됩니다.

# 모델 파일의 경로를 지정합니다.
model_path = "C:/Projects/web/flask_app/best_model_bert.keras"

# 경로에 해당 파일이 존재하는지 확인하고, 그 결과(True/False)를 출력합니다.
print(os.path.exists(model_path))
