import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Giả sử bạn đã load df
file_path = "data/CIDDS-001-Sampled-Data.parquet"
df = pd.read_parquet(file_path)

print("--- 1. TIỀN XỬ LÝ VÀ CHUẨN HÓA DỮ LIỆU ---")
# 1.1. Làm sạch cột 'Bytes'
def clean_bytes(val):
    if pd.isna(val): return 0.0
    val = str(val).strip().upper()
    if 'M' in val: return float(val.replace('M', '')) * 1_000_000
    elif 'K' in val: return float(val.replace('K', '')) * 1_000
    else: return float(val)

df['Bytes'] = df['Bytes'].apply(clean_bytes)

# 1.2. Chọn 10 features cốt lõi
features = [
    'Src IP Addr', 'Src Pt', 'Dst IP Addr', 'Dst Pt', 
    'Proto', 'Flags', 'Tos', 'Duration', 'Bytes', 'Packets'
]
X = df[features].copy()
y = df['attackType'].copy()

# 1.3. Frequency Encoding cho các cột Categorical
cat_cols = X.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
for col in cat_cols:
    freq = X[col].value_counts(normalize=True)
    X[col] = X[col].map(freq).fillna(0)

# 1.4. Scale dữ liệu về [0, 1] và Label Encode nhãn
X = X.astype(float)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


print("\n--- 2. TẠO CHUỖI MULTI-FLOW (WINDOW = 10) ---")
def create_sequences(X_data, y_data, window_size=10):
    X_seq, y_seq = [], []
    # Trượt qua dữ liệu để lấy 10 luồng liên tiếp
    for i in range(len(X_data) - window_size):
        X_seq.append(X_data[i : i + window_size])
        y_seq.append(y_data[i + window_size])
    return np.array(X_seq), np.array(y_seq)

window_size = 10
X_seq, y_seq = create_sequences(X_scaled, y_encoded, window_size)
print(f"Kích thước ma trận 3D tạo thành: {X_seq.shape}")


print("\n--- 3. CHIA TẬP TRAIN/TEST (KHÔNG SHUFFLE) ---")
# QUAN TRỌNG: shuffle=False để bảo toàn trình tự thời gian của chuỗi Multi-flow [cite: 268, 592]
X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
    X_seq, y_seq, test_size=0.3, shuffle=False
)

# Chuyển nhãn sang One-Hot Encoding cho mô hình LSTM
y_train_categorical = tf.keras.utils.to_categorical(y_train_seq)
y_test_categorical = tf.keras.utils.to_categorical(y_test_seq)


print("\n--- 4. HUẤN LUYỆN RANDOM FOREST (MULTI-FLOW BỊ LÀM PHẲNG) ---")
# RF không hiểu được 3D nên phải làm phẳng ma trận (Samples, 10, 10) -> (Samples, 100) [cite: 585, 640]
X_train_flat = X_train_seq.reshape(X_train_seq.shape[0], -1)
X_test_flat = X_test_seq.reshape(X_test_seq.shape[0], -1)

rf_multi = RandomForestClassifier(
    n_estimators=100,            # Cấu hình theo Paper 2 [cite: 434]
    max_depth=35,                # Giới hạn độ sâu để tránh quá khớp [cite: 434]
    max_features='sqrt',         # Dùng căn bậc 2 số đặc trưng [cite: 434]
    class_weight='balanced',     # Paper 2 có dùng class_weight [cite: 431, 434]
    n_jobs=-1,
    random_state=42
)
rf_multi.fit(X_train_flat, y_train_seq)
rf_multi_preds = rf_multi.predict(X_test_flat)

print(f"-> F1-Score của Random Forest Multi-flow: {f1_score(y_test_seq, rf_multi_preds, average='macro'):.4f}")


print("\n--- 5. HUẤN LUYỆN DEEP LEARNING LSTM (MULTI-FLOW NGUYÊN BẢN 3D) ---")
# Cấu hình LSTM theo Bảng 6 của Paper 2 [cite: 577]
lstm_model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(window_size, X_seq.shape[2]), activation='tanh'),
    Dropout(0.2),
    LSTM(100, return_sequences=False, activation='tanh'),
    Dropout(0.2),
    Dense(len(label_encoder.classes_), activation='softmax')
])

lstm_model.compile(
    optimizer=Adam(learning_rate=0.001), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# Chạy với 5 epochs để test (Paper gốc dùng 50 epochs, nhưng 5 là đủ để thấy xu hướng) [cite: 580]
lstm_model.fit(
    X_train_seq, y_train_categorical, 
    epochs=5, 
    batch_size=1024, 
    validation_split=0.1, 
    verbose=1
)

# Đánh giá LSTM
lstm_preds_probs = lstm_model.predict(X_test_seq)
lstm_preds = np.argmax(lstm_preds_probs, axis=1)

print(f"\n-> F1-Score của LSTM Multi-flow: {f1_score(y_test_seq, lstm_preds, average='macro'):.4f}")
print("\n[CHI TIẾT LSTM MULTI-FLOW]")
print(classification_report(y_test_seq, lstm_preds, target_names=label_encoder.classes_, digits=4, zero_division=0))