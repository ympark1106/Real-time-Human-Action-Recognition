import zipfile
import os

zip_path = "C:\\Users\\USER\\Workspace\\HAL\\st_gcn\\nturgbd_skeletons_s001_to_s017.zip"

extract_to = "C:\\Users\\USER\\Workspace\\HAL\\st_gcn\\nturgbd_skeletons"

if not os.path.exists(extract_to):
    os.makedirs(extract_to)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print(f"압축 해제 완료: {extract_to}")

