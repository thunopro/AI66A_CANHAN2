import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, LabelEncoder

# Giả sử df là dữ liệu bạn đã load
# df = pd.read_parquet("data/CIDDS-001-Sampled-Data.parquet")

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

# Cập nhật CHÍNH XÁC tên 10 cột đặc trưng dựa trên file thực tế của bạn
features = [
    'Src IP Addr', 'Src Pt', 'Dst IP Addr', 'Dst Pt', 
    'Proto', 'Flags', 'Tos', 'Duration', 'Bytes', 'Packets'
]

# Tách features (X) và target (y)
X = df[features].copy()
y = df['attackType'].copy()

print("2. Đang chia tập Train/Test (Split 70/30)...")
# Stratify theo y để giữ nguyên tỷ lệ các loại tấn công
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("3. Đang mã hóa các biến phân loại (Categorical Encoding)...")
# Xác định các cột dạng chuỗi (thường là IP, Proto, Flags)
cat_cols = X_train.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

# Fit TRÊN TẬP TRAIN để chống Data Leakage
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
X_test[cat_cols] = encoder.transform(X_test[cat_cols])

# Chuyển toàn bộ X về float để chuẩn bị cho bước Scale
X_train = X_train.astype(float)
X_test = X_test.astype(float)

# Mã hóa Label (attackType)
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

print("4. Đang chuẩn hóa dữ liệu (Min-Max Scaling 0-1)...")
# Fit scaler CHỈ TRÊN TẬP TRAIN
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nHoàn tất! Kích thước dữ liệu:")
print(f"Train set: {X_train_scaled.shape}")
print(f"Test set: {X_test_scaled.shape}")
print(f"Các lớp tấn công đã encode: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")