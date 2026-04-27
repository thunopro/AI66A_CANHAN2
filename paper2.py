import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

print("======================================================")
print("   BẮT ĐẦU CHẠY FULL DATA: CHỈ TEST RANDOM FOREST     ")
print("======================================================")

# ==========================================
# 1. LOAD VÀ TIỀN XỬ LÝ CHUNG
# ==========================================
file_path = "data/CIDDS-001-Sampled-Data.parquet"
df = pd.read_parquet(file_path)
print(f"-> Đã nạp thành công {len(df)} dòng dữ liệu gốc.")

def clean_bytes(val):
    if pd.isna(val): return 0.0
    val = str(val).strip().upper()
    if 'M' in val: return float(val.replace('M', '')) * 1_000_000
    elif 'K' in val: return float(val.replace('K', '')) * 1_000
    else: return float(val)

df['Bytes'] = df['Bytes'].apply(clean_bytes)

features = ['Src IP Addr', 'Src Pt', 'Dst IP Addr', 'Dst Pt', 'Proto', 'Flags', 'Tos', 'Duration', 'Bytes', 'Packets']
X = df[features].copy()
y = df['attackType'].copy()

del df
gc.collect()

# ==========================================
# 2. CHIA TẬP TRAIN/TEST VÀ CHUẨN HÓA
# ==========================================
print("\n-> Đang chia tập Train/Test theo thời gian...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

print("-> Đang thực hiện Frequency Encoding và Min-Max Scaling...")
cat_cols = X_train.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
for col in cat_cols:
    freq = X_train[col].value_counts(normalize=True)
    X_train[col] = X_train[col].map(freq).fillna(0)
    X_test[col] = X_test[col].map(freq).fillna(0)

X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32) 
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# ==========================================
# 3. TẠO CHUỖI MULTI-FLOW TỐI ƯU
# ==========================================
# BẠN CÓ THỂ ĐỔI THÀNH 70 Ở ĐÂY ĐỂ KIỂM CHỨNG SỰ TỤT DỐC CỦA RF
window_rf = 10 
print(f"\n-> Đang tạo ma trận chuỗi 3D cho RF (Window = {window_rf})...")

def create_sequences_optimized(X_data, y_data, window_size):
    n_samples = len(X_data) - window_size + 1
    X_seq = np.zeros((n_samples, window_size, X_data.shape[1]), dtype=np.float32)
    y_seq = np.zeros((n_samples,), dtype=y_data.dtype)
    for i in range(window_size - 1, len(X_data)):
        idx = i - window_size + 1
        X_seq[idx] = X_data[idx : i + 1] 
        y_seq[idx] = y_data[i]           
    return X_seq, y_seq

X_train_seq_rf, y_train_seq_rf = create_sequences_optimized(X_train_scaled, y_train_encoded, window_rf)
X_test_seq_rf, y_test_seq_rf = create_sequences_optimized(X_test_scaled, y_test_encoded, window_rf)

# ==========================================
# 4. HUẤN LUYỆN MÔ HÌNH RANDOM FOREST
# ==========================================
print("\n======================================================")
print(f"   HUẤN LUYỆN RANDOM FOREST (LÀM PHẲNG WINDOW = {window_rf})")
print("======================================================")

# Làm phẳng mảng 3D thành 2D để đưa vào RF
X_train_flat = X_train_seq_rf.reshape(X_train_seq_rf.shape[0], -1)
X_test_flat = X_test_seq_rf.reshape(X_test_seq_rf.shape[0], -1)

# Cấu hình RF chuẩn theo Paper 2
rf_multi = RandomForestClassifier(
    n_estimators=100, 
    max_depth=35, 
    max_features=100, 
    class_weight='balanced', 
    n_jobs=-1, 
    random_state=42
)

print("-> Đang huấn luyện Random Forest... (Vui lòng chờ vài phút)")
rf_multi.fit(X_train_flat, y_train_seq_rf)

print("-> Đang dự đoán tập Test...")
rf_multi_preds = rf_multi.predict(X_test_flat)

# Tìm các nhãn thực sự có trong tập test của chuỗi
labels_in_test_rf = np.unique(y_test_seq_rf)
names_in_test_rf = label_encoder.inverse_transform(labels_in_test_rf)

print(f"\n=> KẾT QUẢ: F1-Score Random Forest (Window {window_rf}): {f1_score(y_test_seq_rf, rf_multi_preds, average='macro'):.4f}")
print(f"\n[BẢNG CHI TIẾT TỪNG LỚP TẤN CÔNG (WINDOW {window_rf})]")
print(classification_report(y_test_seq_rf, rf_multi_preds, labels=labels_in_test_rf, target_names=names_in_test_rf, digits=4, zero_division=0))