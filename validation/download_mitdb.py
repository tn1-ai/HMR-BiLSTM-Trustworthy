# validation/download_mitdb.py
import os
from pathlib import Path
import wfdb

RECORDS = [
    '100', '101', '103', '105', '106', '108', '109', '111', '112', '113',
    '114', '115', '116', '117', '118', '119', '121', '122', '123', '124',
    '200', '201', '202', '203', '205', '207', '208', '209', '210', '212',
    '213', '214', '215', '219', '220', '221', '222', '223', '228', '230',
    '231', '232', '233', '234'
]

def download_all():
    out_dir = Path("data/raw/mitdb")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    missing_records = []
    for r in RECORDS:
        dat_exists = (out_dir / f"{r}.dat").exists()
        hea_exists = (out_dir / f"{r}.hea").exists()
        atr_exists = (out_dir / f"{r}.atr").exists()
        
        if not (dat_exists and hea_exists and atr_exists):
            missing_records.append(r)
            
    if not missing_records:
        print("All 44 MIT-BIH records are already downloaded in data/raw/mitdb.")
        return
        
    print(f"Downloading {len(missing_records)} missing records: {missing_records}")
    wfdb.dl_database("mitdb", str(out_dir), records=missing_records)
    print("Download complete.")

if __name__ == "__main__":
    download_all()
