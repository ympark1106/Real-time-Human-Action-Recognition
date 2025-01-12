import torch
import numpy as np
from st_gcn.st_gcn.net import ST_GCN
from st_gcn.st_gcn.graph import NTU_RGB_D

class STGCNRunner:
    def __init__(self, model_path):
        self.graph = NTU_RGB_D()  # NTU_RGB_D 그래프 초기화
        self.model = ST_GCN(
            channel=3,  # 스켈레톤의 채널: x, y, confidence
            num_class=60,  # 클래스 수 
            window_size=300,  # 프레임 수
            num_point=25,  # 관절 수
            num_person=2,  # 사람 수
            graph='st_gcn.st_gcn.graph.NTU_RGB_D',  
            graph_args={},
        )
        self.model.load_state_dict(torch.load(model_path))
        # self._load_checkpoint(model_path)
        self.model.eval()
        
    def _load_checkpoint(self, model_path):
        checkpoint = torch.load(model_path)

        # 체크포인트 키를 현재 모델 구조에 맞게 매핑
        state_dict = checkpoint
        new_state_dict = {}
        for k, v in state_dict.items():
            if k in self.model.state_dict():
                new_state_dict[k] = v
            else:
                print(f"Skipping {k} as it does not match current model.")

        # 키 불일치 무시
        self.model.load_state_dict(new_state_dict, strict=False)

    def predict_action(self, skeleton_data):
        with torch.no_grad():
            input_tensor = torch.tensor(skeleton_data, dtype=torch.float32)
            output = self.model(input_tensor)
            predicted_label = torch.argmax(output, dim=1).item()
        return predicted_label


    def preprocess_skeleton(self, skeleton_data):
        """
        ST-GCN 입력 데이터 형식으로 skeleton 데이터를 전처리합니다.

        Args:
            skeleton_data: NumPy 배열 형태의 skeleton 데이터 (T, V, C).

        Returns:
            전처리된 Tensor (1, C, T, V, M).
        """
        # skeleton_data는 (T, V, C) 형식이어야 함
        T, V, C = skeleton_data.shape
        if T > self.sequence_length:
            skeleton_data = skeleton_data[-self.sequence_length:, :, :]
        else:
            padding = np.zeros((self.sequence_length - T, V, C))
            skeleton_data = np.concatenate([padding, skeleton_data], axis=0)

        # (T, V, C) -> (C, T, V)
        skeleton_data = np.transpose(skeleton_data, (2, 0, 1))

        # (C, T, V) -> (1, C, T, V, M)
        skeleton_data = np.expand_dims(skeleton_data, axis=(0, -1))

        return torch.tensor(skeleton_data, dtype=torch.float32, device=self.device)

