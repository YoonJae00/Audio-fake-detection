# 딥러닝 기반 오디오 딥페이크 탐지 시스템

## 프로젝트 개요
이 프로젝트는 오디오 파일의 특성(feature)을 분석하여 진짜와 가짜(딥페이크) 오디오를 구분하는 딥러닝 모델을 구현했습니다.

## 데이터셋
- 데이터셋은 진짜 오디오와 가짜 오디오의 특성이 추출된 CSV 파일을 사용
- 각 오디오 파일당 26개의 특성값 포함
- 테스트 데이터셋만 사용하여 모델 학습 및 평가 진행

## 구현 내용

### 1. 데이터 전처리

```40:47:main.py
    df = pd.read_csv(csv_path)
    
    # test 데이터만 필터링
    df = df[df['split'] == 'testing']
    
    # 특성과 레이블 분리
    features = df.iloc[:, 4:].values  # 0-25 특성
    labels = (df['label'] == 'real').astype(int).values
```

- CSV 파일에서 테스트 데이터만 필터링
- 26개의 특성값을 입력 데이터로 사용
- 레이블은 'real'(1)과 'fake'(0)로 이진 분류

### 2. 데이터셋 클래스 구현

```10:19:main.py
class AudioFeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
```

- PyTorch의 Dataset 클래스를 상속하여 커스텀 데이터셋 구현
- 특성값을 FloatTensor로, 레이블을 LongTensor로 변환
- `__getitem__` 메소드로 개별 데이터 접근 가능

### 3. 딥러닝 모델 구조

```22:36:main.py
class DeepFakeDetector(nn.Module):
    def __init__(self, input_size=26):
        super(DeepFakeDetector, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )
    
    def forward(self, x):
        return self.model(x)
```

- 3개의 선형 레이어로 구성된 신경망
- 입력층(26) → 은닉층1(64) → 은닉층2(32) → 출력층(2)
- ReLU 활성화 함수와 드롭아웃 레이어 사용
- 과적합 방지를 위한 드롭아웃 비율: 0.3, 0.2

### 4. 학습 프로세스

```68:80:main.py
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
```

- CrossEntropyLoss 손실 함수 사용
- Adam 옵티마이저로 모델 파라미터 최적화
- 미니배치 단위로 학습 진행
- GPU 가속 지원 (CUDA 사용 가능 시)

### 5. 성능 평가

```83:104:main.py
        if (epoch + 1) % 1 == 0:
            model.eval()
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for features, labels in test_loader:
                    features, labels = features.to(device), labels.to(device)
                    outputs = model(features)
                    _, predicted = torch.max(outputs.data, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            accuracy = accuracy_score(all_labels, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
            
            print(f'\n에포크 {epoch + 1}/{epochs}:')
            print(f'손실: {total_loss/len(train_loader):.4f}')
            print(f'정확도: {accuracy:.4f}')
            print(f'정밀도: {precision:.4f}')
            print(f'재현율: {recall:.4f}')
            print(f'F1 점수: {f1:.4f}')
```

- 매 에포크마다 모델 성능 평가
- 평가 지표:
  - 정확도 (Accuracy)
  - 정밀도 (Precision)
  - 재현율 (Recall)
  - F1 점수

## 사용 방법
1. 필요한 패키지 설치:
```bash
pip install torch pandas scikit-learn numpy
```

2. 모델 학습 실행:
```python
python main.py
```

## 기술 스택
- Python 3.12
- PyTorch
- pandas
- scikit-learn
- numpy

## 향후 개선 사항
- 모델 아키텍처 최적화
- 하이퍼파라미터 튜닝
- 데이터 증강 기법 적용
- 모델 저장 및 로드 기능 추가
- 실시간 오디오 분석 기능 구현
