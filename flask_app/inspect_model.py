# 이 코드는 'best_model_bert.keras' 파일이 존재하는지 확인하고, 존재하면
# 해당 파일이 HDF5 형식인지 확인하는 코드입니다.
# 주로 모델 파일의 형식을 확인하거나 파일을 읽을 때 사용됩니다.

import os  # os 모듈은 파일 시스템 작업을 처리하는 데 사용됩니다.

# 확인할 모델 파일 경로를 지정합니다.
model_path = "best_model_bert.keras"

# 파일이 존재하는지 확인합니다.
if os.path.exists(model_path):
    print(f"File {model_path} exists.")  # 파일이 존재하면 메시지를 출력합니다.
    
    try:
        # 파일을 바이너리 모드("rb")로 열어서 첫 8바이트를 읽습니다.
        with open(model_path, "rb") as f:
            header = f.read(8)  # 파일의 첫 8바이트를 읽어 header에 저장합니다.
            print(f"File header: {header}")  # 읽은 파일 헤더를 출력합니다.
            
            # 헤더가 HDF5 파일 서명(b"\x89HDF")으로 시작하는지 확인합니다.
            if header.startswith(b"\x89HDF"):
                print("File appears to be an HDF5 file.")  # HDF5 파일로 판단되면 메시지를 출력합니다.
            else:
                print("File is not an HDF5 file.")  # 그렇지 않으면 HDF5 파일이 아님을 출력합니다.
    except Exception as e:
        # 파일을 열거나 처리 중 오류가 발생하면 예외를 처리하고 오류 메시지를 출력합니다.
        print(f"Error opening file: {e}")
else:
    print(f"File {model_path} does not exist.")  # 파일이 존재하지 않으면 메시지를 출력합니다.
