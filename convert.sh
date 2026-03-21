#!/bin/bash
mkdir -p CIDDS-001/traffic/OpenStack_Parquet

for f in CIDDS-001/traffic/OpenStack/*.csv; do
    filename=$(basename "$f" .csv)
    echo "Đang chuyển: $filename"
    duckdb -c "COPY (SELECT * FROM '$f') TO 'CIDDS-001/traffic/OpenStack_Parquet/$filename.parquet' (FORMAT PARQUET);"
done