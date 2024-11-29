

# 시계열 기반 오디오 딥페이크 탐지 시스템

## 프로젝트 개요
이 프로젝트는 LSTM(Long Short-Term Memory) 네트워크를 사용하여 오디오의 시계열 특성을 분석하고, 진짜와 가짜(딥페이크) 오디오를 구분하는 딥러닝 모델을 구현했습니다.

## 데이터셋
- 오디오 파일의 26개 특성이 추출된 CSV 파일 사용
- 각 특성의 시계열 패턴을 분석하여 분류 수행
- 테스트 데이터셋을 사용하여 모델 학습 및 평가

## 구현 내용

### 1. 시계열 데이터 전처리

```python
class AudioFeatureDataset(Dataset):
    def __init__(self, features, labels, sequence_length=10):
        self.sequence_length = sequence_length
        self.sequences = []
        self.labels = []
        
        for i in range(len(features) - sequence_length + 1):
            self.sequences.append(features[i:i + sequence_length])
            self.labels.append(labels[i])
```

- 연속된 특성값들을 시퀀스로 구성
- 시퀀스 길이(sequence_length) 파라미터로 조절 가능
- 슬라이딩 윈도우 방식으로 시계열 데이터 생성

### 2. LSTM 기반 모델 구조

```python
class DeepFakeDetector(nn.Module):
    def __init__(self, input_size=26, hidden_size=64, num_layers=2, sequence_length=10):
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * sequence_length, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
```

#### 모델 구조:
1. **LSTM 레이어**
   - 입력 크기: 26 (특성 수)
   - 은닉층 크기: 64
   - 레이어 수: 2
   - 드롭아웃: 0.3

2. **완전연결층**
   - LSTM 출력을 평탄화
   - 64 유닛의 은닉층
   - 드롭아웃: 0.2
   - 2개의 출력 클래스 (진짜/가짜)

### 3. 학습 프로세스

```python
def train_model(csv_path, epochs=10, batch_size=32, sequence_length=10):
    # 데이터 로딩 및 전처리
    df = pd.read_csv(csv_path)
    df = df[df['split'] == 'testing']
    
    # 특성과 레이블 분리
    features = df.iloc[:, 4:].values
    labels = (df['label'] == 'real').astype(int).values
```

- CrossEntropyLoss 손실 함수 사용
- Adam 옵티마이저로 최적화
- 미니배치 학습
- GPU 가속 지원

### 4. 성능 평가 지표
- 정확도 (Accuracy)
- 정밀도 (Precision)
- 재현율 (Recall)
- F1 점수

![image](https://github.com/user-attachments/assets/5b67a4e6-2770-4446-9735-a2f00a801a09)


![image](https://github.com/user-attachments/assets/7902834e-68d0-4ca1-8fd2-06223b275bac)


## 설치 및 실행 방법

1. 필요한 패키지 설치:
```bash
pip install torch pandas scikit-learn numpy
```

2. 모델 학습 실행:
```python
python main.py
```

3. 시퀀스 길이 조정:
```python
model = train_model(csv_path, sequence_length=15)  # 기본값: 10
```

## 기술 스택
- Python 3.12
- PyTorch
- pandas
- scikit-learn
- numpy

## 주요 특징
- LSTM을 활용한 시계열 패턴 학습
- 가변적인 시퀀스 길이 설정 가능
- 다층 LSTM 구조로 복잡한 패턴 포착
- 드롭아웃을 통한 과적합 방지

## 향후 개선 사항
- 양방향 LSTM (Bidirectional LSTM) 적용
- 어텐션 메커니즘 도입
- 시퀀스 길이 최적화
- 데이터 증강 기법 적용
- 모델 저장 및 로드 기능
- 실시간 오디오 분석 기능

## 참고 사항
- 시퀀스 길이가 길수록 더 많은 컨텍스트를 고려할 수 있지만, 메모리 사용량이 증가합니다.
- GPU 메모리 한계를 고려하여 배치 크기와 시퀀스 길이를 조절하세요.
- 테스트 데이터만 사용하므로, 전체 데이터셋으로 확장하여 성능을 개선할 수 있습니다.

## 라이센스
이 프로젝트는 MIT 라이센스를 따릅니다.