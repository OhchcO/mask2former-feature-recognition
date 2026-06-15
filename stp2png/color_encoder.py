# -*- coding: utf-8 -*-
"""
颜色编码器：面ID + 类型 + 面积 → RGB 颜色

编码方案：
  R = 类型基值 + 面积偏移（间隙=51，5种类型均分0-254）
  G, B = 面ID自适应网格分配

用法：
    from color_encoder import FaceColorEncoder

    encoder = FaceColorEncoder(num_faces=50, shuffle=True, seed=42)

    # 编码
    rgb = encoder.encode(face_id=1, type_id=0, area_ratio=0.5)
    # rgb = (25, 152, 0)

    # 解码
    result = encoder.decode(rgb)
    # result = {"face_id": 1, "type_id": 0, "area_ratio": 0.49}
"""
import json
import math
import numpy as np


# 类型基值：5种类型均分 0-254，间隙=51
TYPE_GAP = 51
TYPE_R_BASE = {
    0: 0,     # 平面:    R ∈ [0,  50]
    1: 51,    # 圆柱面:  R ∈ [51, 101]
    2: 102,   # 圆锥面:  R ∈ [102,152]
    3: 153,   # 球面:    R ∈ [153,203]
    4: 204,   # 其他面:  R ∈ [204,254]
}

TYPE_NAMES = {
    0: "平面",
    1: "圆柱面",
    2: "圆锥面",
    3: "球面",
    4: "其他面",
}


class FaceColorEncoder:
    """面ID颜色编码器"""

    def __init__(self, num_faces, shuffle=True, seed=42):
        """
        Args:
            num_faces: 面数量
            shuffle: 是否打乱GB分配（让相邻ID颜色差异更大）
            seed: 随机种子
        """
        self.num_faces = num_faces
        self.shuffle = shuffle
        self.seed = seed
        self.gb_mapping = self._compute_gb_mapping()
        self.gb_reverse = {v: k for k, v in self.gb_mapping.items()}

    def _compute_gb_mapping(self):
        """自适应网格分配面ID → (G, B)"""
        K = math.ceil(math.sqrt(self.num_faces))
        step = 254 / K
        grid_coords = []
        for i in range(self.num_faces):
            row = i // K
            col = i % K
            g = min(int(row * step), 254)
            b = min(int(col * step), 254)
            grid_coords.append((g, b))
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(grid_coords)
        return {i + 1: grid_coords[i] for i in range(self.num_faces)}

    def encode(self, face_id, type_id, area_ratio):
        """编码：面ID + 类型 + 面积 → RGB

        Args:
            face_id: 面ID（1开始）
            type_id: 类型ID（0-4）
            area_ratio: 归一化面积（0.0~1.0）

        Returns:
            (R, G, B) 三元组，值域 0-254
        """
        g, b = self.gb_mapping[face_id]
        base = TYPE_R_BASE[type_id]
        offset = min(int(area_ratio * TYPE_GAP), TYPE_GAP - 1)
        r = base + offset
        return (r, g, b)

    def decode(self, rgb):
        """解码：RGB → 面ID + 类型 + 面积

        Args:
            rgb: (R, G, B) 三元组

        Returns:
            dict: {"face_id", "type_id", "area_ratio"}
        """
        r, g, b = rgb[0], rgb[1], rgb[2]
        type_id = r // TYPE_GAP
        area_offset = r % TYPE_GAP
        area_ratio = round(area_offset / TYPE_GAP, 4)
        face_id = self.gb_reverse.get((g, b), None)
        return {
            "face_id": face_id,
            "type_id": type_id,
            "area_ratio": area_ratio,
        }

    def encode_face_list(self, faces):
        """批量编码

        Args:
            faces: [{"face_id": 1, "type_id": 0, "area_ratio": 0.5}, ...]

        Returns:
            dict: {face_id: (R, G, B)}
        """
        return {
            f["face_id"]: self.encode(f["face_id"], f["type_id"], f["area_ratio"])
            for f in faces
        }

    def save_mapping(self, path, extra_config=None):
        """保存映射表到 JSON"""
        mapping = {
            "faces": {
                str(fid): {"G": g, "B": b}
                for fid, (g, b) in self.gb_mapping.items()
            },
            "config": {
                "num_faces": self.num_faces,
                "shuffle": self.shuffle,
                "seed": self.seed,
                "K": math.ceil(math.sqrt(self.num_faces)),
                "type_gap": TYPE_GAP,
                "type_r_base": TYPE_R_BASE,
            }
        }
        if extra_config:
            mapping["config"].update(extra_config)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_mapping(cls, path):
        """从 mapping.json 加载（不解码，只加载GB映射）"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = data["config"]
        encoder = cls(
            num_faces=cfg["num_faces"],
            shuffle=cfg.get("shuffle", False),
            seed=cfg.get("seed", 42),
        )
        return encoder


def type_area_to_r(type_id, area_ratio):
    """独立函数：类型+面积 → R值"""
    base = TYPE_R_BASE[type_id]
    offset = min(int(area_ratio * TYPE_GAP), TYPE_GAP - 1)
    return base + offset


def r_to_type_area(r_value):
    """独立函数：R值 → (类型ID, 面积偏移)"""
    type_id = r_value // TYPE_GAP
    area_offset = r_value % TYPE_GAP
    return type_id, area_offset
