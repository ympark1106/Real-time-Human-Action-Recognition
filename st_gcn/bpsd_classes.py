import json

# 입력 JSON 경로
input_json_path = "/SSDe/youmin_park/Real-time-Human-Action-Recognition/st_gcn/data/Kinetics/kinetics-skeleton/kinetics_val_label.json"

# 출력 JSON 경로
output_json_path = "kinetics_val_label_bpsd.json"

# BPSD 클래스 정의
bpsd_labels = ["normal", "aggression", "tumbles"]
label_to_index = {label: idx for idx, label in enumerate(bpsd_labels)}

# 기존 라벨 정의
normal_labels = [
    "Applauding", "Arranging flowers", "Answering questions", "Assembling computer",
    "Baking cookies", "Biking through snow", "Braiding hair", "Brushing hair",
    "Brushing teeth", "Cleaning floor", "Cleaning windows", "Cooking",
    "Counting money", "Drinking", "Eating", "Folding clothes", "Jogging",
    "Knitting", "Laughing", "Making", "Petting animal", "Reading book",
    "Walking the dog", "Washing dishes", "Washing hair", "Washing hands",
    "Writing"
]

aggression_labels = [
    "Punching bag", "Punching person", "Wrestling", "Headbutting",
    "Slapping", "Shoving", "Choking"
]

tumbles_labels = [
    "Falling down", "Tripping", "Faceplanting"
]

# 라벨을 소문자로 변환
normal_labels_lower = [label.lower() for label in normal_labels]
aggression_labels_lower = [label.lower() for label in aggression_labels]
tumbles_labels_lower = [label.lower() for label in tumbles_labels]

# JSON 데이터를 필터링하여 라벨 및 인덱스 변경
with open(input_json_path, 'r') as infile:
    data = json.load(infile)

# 결과를 저장할 딕셔너리
filtered_data = {}
for key, value in data.items():
    label_lower = value["label"].lower()
    if label_lower in normal_labels_lower:
        filtered_data[key] = {
            "label": "normal",  # bpsd 레이블로 변경
            "has_skeleton": value["has_skeleton"],
            "label_index": label_to_index["normal"]  # bpsd 인덱스로 변경
        }
    elif label_lower in aggression_labels_lower:
        filtered_data[key] = {
            "label": "aggression",  # bpsd 레이블로 변경
            "has_skeleton": value["has_skeleton"],
            "label_index": label_to_index["aggression"]  # bpsd 인덱스로 변경
        }
    elif label_lower in tumbles_labels_lower:
        filtered_data[key] = {
            "label": "tumbles",  # bpsd 레이블로 변경
            "has_skeleton": value["has_skeleton"],
            "label_index": label_to_index["tumbles"]  # bpsd 인덱스로 변경
        }
    else:
        print(f"Warning: Label '{value['label']}' for sample '{key}' is not mapped to any category.")

# 필터링된 데이터 저장
with open(output_json_path, 'w') as outfile:
    json.dump(filtered_data, outfile, indent=4)

print(f"Filtered data saved to {output_json_path}")
