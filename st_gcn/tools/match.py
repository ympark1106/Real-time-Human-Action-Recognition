import os
import json

# data_path = "./data/Kinetics/kinetics-skeleton/kinetics_val"
# label_path = "./data/Kinetics/kinetics-skeleton/kinetics_val_label_bpsd.json"
data_path = "./data/Kinetics/kinetics-skeleton/kinetics_train"
label_path = "./data/Kinetics/kinetics-skeleton/kinetics_train_label_bpsd.json"

'''
# Load JSON labels
with open(label_path, 'r') as f:
    labels = json.load(f)

# Get sample IDs from JSON and data_path
json_ids = set(labels.keys())
data_ids = set([os.path.splitext(file)[0] for file in os.listdir(data_path)])

# Find mismatched IDs
missing_in_data = json_ids - data_ids
missing_in_json = data_ids - json_ids

matching_in_json = data_ids & json_ids

print(f"Total samples in JSON: {len(json_ids)}")
print(f"Total samples in data path: {len(data_ids)}")
print(f"Samples missing in data_path: {len(missing_in_data)}")
print(f"Samples missing in JSON: {len(missing_in_json)}")

# Optionally, print the mismatched IDs
print("Examples missing in data_path:", list(missing_in_data)[:10])
print("Examples missing in JSON:", list(missing_in_json)[:10])

print("Examples matching in both JSON and data_path:", list(matching_in_json)[:10])
print("Total samples matching in both JSON and data_path:", len(matching_in_json))
'''


# JSON 레이블 로드
with open(label_path, 'r') as f:
    labels = json.load(f)

# 데이터 디렉토리에서 파일명 추출
data_files = set([os.path.splitext(file)[0] for file in os.listdir(data_path)])

# JSON에서 레이블 키 추출
json_keys = set(labels.keys())

# 매칭된 파일 출력
matching_files = data_files & json_keys
print(f"Total matching files: {len(matching_files)}")
print("Matching files:")
# for file in list(matching_files):
#     print(f"File: {file}, Label: {labels[file]['label']}")
    
#전체 매칭된 숫자
print(f"Total matching files: {len(matching_files)}")

# # 불일치 항목 출력
# data_only = data_files - json_keys
# json_only = json_keys - data_files

# print("\nFiles in data path but not in JSON:")
# for file in list(data_only)[:10]:
#     print(file)

# print("\nFiles in JSON but not in data path:")
# for file in list(json_only)[:10]:
#     print(file)
