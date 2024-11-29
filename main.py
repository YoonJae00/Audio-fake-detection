import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# 데이터셋 클래스 정의
class AudioFeatureDataset(Dataset):
    def __init__(self, features, labels, sequence_length=10):
        # 시계열 시퀀스로 데이터 재구성
        self.sequence_length = sequence_length
        self.sequences = []
        self.labels = []
        
        for i in range(len(features) - sequence_length + 1):
            self.sequences.append(features[i:i + sequence_length])
            self.labels.append(labels[i])
            
        self.sequences = torch.FloatTensor(self.sequences)
        self.labels = torch.LongTensor(self.labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# 모델 정의
class DeepFakeDetector(nn.Module):
    def __init__(self, input_size=26, hidden_size=64, num_layers=2, sequence_length=10):
        super(DeepFakeDetector, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
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
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        
        # LSTM 출력
        lstm_out, _ = self.lstm(x)
        
        # LSTM 출력을 펼치기
        lstm_out = lstm_out.reshape(batch_size, -1)
        
        # 완전연결층 통과
        output = self.fc(lstm_out)
        return output

def train_model(csv_path, epochs=10, batch_size=32, sequence_length=10):
    # CSV 파일 읽기
    df = pd.read_csv(csv_path)
    
    # test 데이터만 필터링
    df = df[df['split'] == 'testing']
    
    # 특성과 레이블 분리
    features = df.iloc[:, 4:].values  # 0-25 특성
    labels = (df['label'] == 'real').astype(int).values
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    # 데이터셋 생성
    train_dataset = AudioFeatureDataset(X_train, y_train, sequence_length)
    test_dataset = AudioFeatureDataset(X_test, y_test, sequence_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 모델, 손실함수, 옵티마이저 초기화
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepFakeDetector(sequence_length=sequence_length).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # 학습
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 평가
        if (epoch + 1) % 1 == 0:
            model.eval()
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for sequences, labels in test_loader:
                    sequences, labels = sequences.to(device), labels.to(device)
                    outputs = model(sequences)
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
    
    return model

if __name__ == "__main__":
    csv_path = "features4.csv"
    model = train_model(csv_path, sequence_length=10)