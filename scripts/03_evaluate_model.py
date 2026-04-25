"""纯脚本版本：评估训练好的人脸检测模型。"""

from __future__ import annotations

import argparse
from pathlib import Path



def parse_args() -> argparse.Namespace:
    """中文注释：解析评估参数。"""
    parser = argparse.ArgumentParser(description="评估 YOLOv8 人脸检测模型")
    parser.add_argument("--model", type=Path, default=Path("models/best.pt"), help="待评估模型路径")
    parser.add_argument("--data", type=Path, default=Path("data/face.yaml"), help="YOLO 数据配置文件")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="评估数据集划分")
    parser.add_argument("--imgsz", type=int, default=640, help="评估尺寸")
    parser.add_argument("--device", type=str, default=None, help="评估设备，例如 cpu 或 0")
    return parser.parse_args()


def main() -> None:
    """中文注释：执行评估并打印关键指标。"""
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"未找到模型文件: {args.model}")
    if not args.data.exists():
        raise FileNotFoundError(f"未找到数据配置文件: {args.data}")

    # 中文注释：延迟导入 ultralytics，便于在未安装依赖时查看 --help
    from ultralytics import YOLO

    model = YOLO(str(args.model))

    val_kwargs = {
        "data": str(args.data),
        "split": args.split,
        "imgsz": args.imgsz,
    }
    if args.device is not None:
        val_kwargs["device"] = args.device

    metrics = model.val(**val_kwargs)

    # 中文注释：从 box 指标中读取 precision/recall/mAP
    precision = float(metrics.box.p)
    recall = float(metrics.box.r)
    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)

    print("评估完成：")
    print(f"Precision : {precision:.6f}")
    print(f"Recall    : {recall:.6f}")
    print(f"mAP50     : {map50:.6f}")
    print(f"mAP50-95  : {map5095:.6f}")


if __name__ == "__main__":
    main()
