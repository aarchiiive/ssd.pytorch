import time
import subprocess

methods = ['dark', 'GLARE'] # lambda3
# methods = ['SCI', 'ZeroDCE', 'ZeroDCE++'] # lambda5
# methods = ['RUAS'] # lambda1
# methods = ['Retinexformer'] # lambda2

for method in methods:
    while True:  # 같은 loss_weight에 대해 무한 반복
        try:
            process = subprocess.run(
                ['python3', 'train.py', '--method', method],
                check=True  # 실패 시 예외 발생
            )
            break  # 정상 종료 시 while 루프 탈출
        except subprocess.CalledProcessError as e:
            print(f"[Error] train.py 실행 실패 (코드 {e.returncode}). 5초 후 재시작...")
            time.sleep(5)
        except Exception as e:
            print(f"[Critical Error] train.py 실행 중 예기치 않은 오류 발생: {e}. 5초 후 재시작...")
            time.sleep(5)