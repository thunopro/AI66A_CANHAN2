import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
import xgboost as xgb

# ==========================================
# PHẦN 1: TẢI VÀ LÀM SẠCH DỮ LIỆU
# ==========================================
file_path = "data/CIDDS-001-Sampled-Data.parquet"
print(f"1. Đang tải dữ liệu từ {file_path}...")
# df = pd.read_parquet(file_path) # Bỏ comment dòng này khi chạy thật

print("2. Đang làm sạch cột 'Bytes'...")
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

# (Giả sử df đã được load)
df['Bytes'] = df['Bytes'].apply(clean_bytes)

features = [
    'Src IP Addr', 'Src Pt', 'Dst IP Addr', 'Dst Pt', 
    'Proto', 'Flags', 'Tos', 'Duration', 'Bytes', 'Packets'
]

X = df[features].copy()
y = df['attackType'].copy()

print("3. Đang chia tập Train/Test (Split 70/30)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ==========================================
# PHẦN 2: FREQUENCY ENCODING & CHUẨN HÓA
# ==========================================
print("4. Đang mã hóa các biến phân loại (Frequency Encoding)...")
cat_cols = X_train.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

# Tính toán tần suất TRÊN TẬP TRAIN và map cho cả Train/Test
for col in cat_cols:
    freq = X_train[col].value_counts(normalize=True)
    X_train[col] = X_train[col].map(freq).fillna(0) # Điền 0 nếu tập Test có IP/Giao thức lạ chưa từng gặp
    X_test[col] = X_test[col].map(freq).fillna(0)

# Chuyển toàn bộ về float
X_train = X_train.astype(float)
X_test = X_test.astype(float)

print("5. Đang mã hóa nhãn (Label Encoding)...")
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

print("6. Đang chuẩn hóa dữ liệu (Min-Max Scaling 0-1)...")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nHoàn tất Tiền xử lý! Kích thước dữ liệu:")
print(f"Train set: {X_train_scaled.shape}")
print(f"Test set: {X_test_scaled.shape}")
print(f"Các lớp tấn công đã encode: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}\n")

# ==========================================
# PHẦN 3: HUẤN LUYỆN MÔ HÌNH
# ==========================================
print("--- ĐANG TRAIN MÔ HÌNH RANDOM FOREST ---")
rf_model = RandomForestClassifier(
    n_estimators=10,             
    criterion='gini',            
    min_samples_split=2,         
    min_samples_leaf=1,          
    max_features='sqrt',         
    random_state=42,             
    n_jobs=-1                    
)
rf_model.fit(X_train_scaled, y_train)
rf_preds = rf_model.predict(X_test_scaled)

print("--- ĐANG TRAIN MÔ HÌNH XGBOOST ---")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,            
    learning_rate=0.1,           
    max_depth=6,                 
    tree_method='hist',          # Vẫn giữ thuật toán siêu tốc
    random_state=42,
    n_jobs=-1                    
)
xgb_model.fit(X_train_scaled, y_train)
xgb_preds = xgb_model.predict(X_test_scaled)

# ==========================================
# PHẦN 4: ĐÁNH GIÁ KẾT QUẢ
# ==========================================
print("\n================ SO SÁNH HIỆU SUẤT (MACRO F1) ================")
print(f"Random Forest F1-Score: {f1_score(y_test, rf_preds, average='macro'):.4f}")
print(f"XGBoost F1-Score:       {f1_score(y_test, xgb_preds, average='macro'):.4f}")
print("==============================================================")

print("\n--- CHI TIẾT BÁO CÁO TỪ XGBOOST ---")
print(classification_report(y_test, xgb_preds, target_names=label_encoder.classes_, digits=4, zero_division=0))