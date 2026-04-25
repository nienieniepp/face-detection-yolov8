"""纯脚本版本：训练 YOLOv8n 人脸检测模型。"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path



def parse_args() -> argparse.Namespace:
    """中文注释：解析训练参数。"""
    parser = argparse.ArgumentParser(description="训练 YOLOv8n 人脸检测模型")
    parser.add_argument("--data", type=Path, default=Path("data/face.yaml"), help="YOLO 数据配置文件")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="训练图片尺寸")
    parser.add_argument("--batch", type=int, default=16, help="批大小")
    parser.add_argument("--device", type=str, default=None, help="训练设备，例如 cpu 或 0")
    parser.add_argument("--project", type=Path, default=Path("runs"), help="训练输出目录")
    parser.add_argument("--name", type=str, default="face_yolov8n", help="训练任务名")
    parser.add_argument("--save-best", type=Path, default=Path("models/best.pt"), help="最佳模型保存路径")
    return parser.parse_args()


def main() -> None:
    """中文注释：执行训练并导出最佳模型。"""
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"未找到数据配置文件: {args.data}")

    # 中文注释：延迟导入 ultralytics，便于在未安装依赖时查看 --help
    from ultralytics import YOLO

    # 中文注释：加载 YOLOv8n 预训练权重
    model = YOLO("yolov8n.pt")

    # 中文注释：启动训练
    train_kwargs = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(args.project),
        "name": args.name,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)

    # 中文注释：优先从训练结果中获取 best.pt 路径
    save_dir = Path(getattr(results, "save_dir", args.project / args.name))
    best_src = save_dir / "weights" / "best.pt"

    if not best_src.exists():
        raise FileNotFoundError(f"训练完成但未找到 best.pt: {best_src}")

    args.save_best.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_src, args.save_best)

    print(f"训练完成，最佳模型已保存到: {args.save_best}")


if __name__ == "__main__":
    main()
