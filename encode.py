import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, LabelEncoder

df = pd.read_parquet("CIDDS-001-Sampled-Data.parquet")

print("1. Đang làm sạch cột 'Bytes'...")
def clean_bytes(val):
    if pd.isna(val):
        return 0.0
    val = str(val).strip().upper()
    if 'M' in val:
        return float(val.replace('M', '')) * 1_000_000
    elif 'K' in val:
        return float(val.replace('K', '')) * 1_000
    else:
        return float(val)

df['Bytes'] = df['Bytes'].apply(clean_bytes)

# Khai báo features (X) và target (y)
features = [
    'Src IP Addr', 'Src Pt', 'Dst IP Addr', 'Dst Pt', 
    'Proto', 'Flags', 'Duration', 'Bytes', 'Packets'
]
X = df[features].copy()
y = df['attackType'].copy()

print("2. Đang chia tập Train/Test (Split 70/30)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("3. Đang mã hóa Đặc trưng (Frequency) và Nhãn (Label)...")
# A. Frequency Encoding cho X (Đầu vào)
cat_cols = X_train.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
for col in cat_cols:
    freq = X_train[col].value_counts(normalize=True)
    X_train[col] = X_train[col].map(freq).fillna(0)
    X_test[col] = X_test[col].map(freq).fillna(0)

# Ép kiểu float chuẩn bị cho Scaling
X_train = X_train.astype(float)
X_test = X_test.astype(float)

# B. Label Encoding cho y (Đầu ra)
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

print("4. Đang chuẩn hóa dữ liệu (Min-Max Scaling 0-1)...")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)