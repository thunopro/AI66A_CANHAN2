import pandas as pd
import os
import pyarrow as pa
import pyarrow.parquet as pq

# 1. Cấu hình đường dẫn
input_folder = 'CIDDS-001/traffic/OpenStack' 
output_folder = 'CIDDS-001/traffic/OpenStack_Parquet'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 2. Duyệt qua từng file CSV
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        csv_path = os.path.join(input_folder, filename)
        parquet_path = os.path.join(output_folder, filename.replace('.csv', '.parquet'))
        
        print(f"Đang chuyển đổi: {filename}...")
        
        # Đọc theo chunk
        reader = pd.read_csv(csv_path, chunksize=500000, low_memory=False)
        
        writer = None
        
        for chunk in reader:
            # Chuyển đổi toàn bộ cột Object (chuỗi) sang String thuần túy để tránh lỗi kiểu dữ liệu
            for col in chunk.select_dtypes(['object']).columns:
                chunk[col] = chunk[col].astype(str)
            
            # Chuyển DataFrame chunk thành Bảng của PyArrow
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            
            # Nếu là chunk đầu tiên, khởi tạo ParquetWriter
            if writer is None:
                writer = pq.ParquetWriter(parquet_path, table.schema, compression='snappy')
            
            # Ghi chunk vào file
            writer.write_table(table)
        
        # Đóng writer sau khi xong một file CSV
        if writer:
            writer.close()
            
        print(f"Hoàn thành: {parquet_path}")

print("\n--- TẤT CẢ ĐÃ CHUYỂN SANG PARQUET XONG! ---")