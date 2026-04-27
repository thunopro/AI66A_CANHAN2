import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score

# ==========================================
# PHẦN 1: TIỀN XỬ LÝ DỮ LIỆU (PREPROCESSING)
# ==========================================

# Giả sử df là dữ liệu bạn đã load
df = pd.read_parquet("data/CIDDS-001-Sampled-Data.parquet")

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

print("\n[+] Hoàn tất Tiền xử lý! Kích thước dữ liệu:")
print(f" - Train set: {X_train_scaled.shape}")
print(f" - Test set: {X_test_scaled.shape}")
print(f" - Từ điển Nhãn (y): {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")


# ==========================================
# PHẦN 2: HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH
# ==========================================
print("\n--- 5. ĐANG TRAIN LẠI MÔ HÌNH ---")

# Random Forest: Đã tăng n_estimators và thêm class_weight='balanced'
rf_model_tuned = RandomForestClassifier(
    n_estimators=10,            # Tăng từ 10 lên 100 cây để chống nhiễu
    criterion='entropy',            
    min_samples_split=2,         
    min_samples_leaf=1,          
    max_features='sqrt',         
    class_weight=None,     # Cứu cánh cho các lớp thiểu số
    random_state=42,             
    n_jobs=-1                    
)
rf_model_tuned.fit(X_train_scaled, y_train)
rf_preds_tuned = rf_model_tuned.predict(X_test_scaled)

# KNN giữ nguyên tham số
knn_model_tuned = KNeighborsClassifier(
    n_neighbors=3, weights='distance', leaf_size=30, metric='minkowski', n_jobs=-1
)
knn_model_tuned.fit(X_train_scaled, y_train)
knn_preds_tuned = knn_model_tuned.predict(X_test_scaled)

print("\n--- KẾT QUẢ SAU KHI TINH CHỈNH (MACRO F1) ---")
print(f"Random Forest F1-Score: {f1_score(y_test, rf_preds_tuned, average='macro'):.4f}")
print(f"KNN F1-Score:           {f1_score(y_test, knn_preds_tuned, average='macro'):.4f}")

print("\n--- CHI TIẾT BÁO CÁO CỦA RANDOM FOREST ---")
print(classification_report(y_test, rf_preds_tuned, target_names=label_encoder.classes_, digits=4, zero_division=0))