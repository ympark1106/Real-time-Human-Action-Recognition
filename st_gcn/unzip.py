import zipfile
import os

zip_path = "/SSDe/youmin_park/Real-time-Human-Action-Recognition/st_gcn/kinetics-skeleton.zip"

extract_to = "/SSDe/youmin_park/Real-time-Human-Action-Recognition/st_gcn/data/KINETICS/"

if not os.path.exists(extract_to):
    os.makedirs(extract_to)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print(f"압축 해제 완료: {extract_to}")

