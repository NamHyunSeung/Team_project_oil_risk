#!/bin/bash
# Oracle Cloud Ubuntu 22.04 ARM 초기 환경 설정
set -e

echo "=== 시스템 패키지 설치 ==="
sudo apt-get update -y
sudo apt-get install -y \
    python3 python3-pip python3-venv git \
    fonts-nanum fonts-nanum-coding \
    libgomp1 libopenblas-dev \
    curl wget

echo "=== 한국어 폰트 캐시 갱신 ==="
fc-cache -fv

echo "=== Python 가상환경 생성 ==="
cd ~/oil_risk
python3 -m venv venv
source venv/bin/activate

echo "=== PyTorch ARM 설치 (CPU) ==="
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "=== 나머지 패키지 설치 ==="
pip install -r requirements.txt

echo "=== FinBERT 모델 사전 다운로드 (~1.4GB) ==="
python3 -c "
from transformers import pipeline
print('FinBERT 다운로드 중...')
pipeline('text-classification', model='ProsusAI/finbert')
print('완료')
"

echo "=== 출력 디렉터리 생성 ==="
mkdir -p output logs data

echo ""
echo "✅ 설치 완료. 다음 명령어로 파이프라인 실행:"
echo "   source ~/oil_risk/venv/bin/activate"
echo "   python oil_risk_mvp.py"
