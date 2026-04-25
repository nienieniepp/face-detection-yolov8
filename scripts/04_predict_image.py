"""纯脚本版本：对单张图片或图片目录进行人脸检测。"""

from __future__ import annotations

import argparse
from pathlib import Path



def parse_args() -> argparse.Namespace:
    """中文注释：解析图片预测参数。"""
    parser = argparse.ArgumentParser(description="YOLOv8 人脸检测（图片）")
    parser.add_argument("--model", type=Path, default=Path("models/best.pt"), help="模型路径")
    parser.add_argument("--source", type=Path, required=True, help="输入图片或图片目录")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    parser.add_argument("--device", type=str, default=None, help="推理设备，例如 cpu 或 0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/image_predictions"),
        help="输出目录",
    )
    return parser.parse_args()


def main() -> None:
    """中文注释：执行图片预测并保存结果。"""
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"未找到模型文件: {args.model}")
    if not args.source.exists():
        raise FileNotFoundError(f"未找到输入源: {args.source}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 中文注释：延迟导入 ultralytics，便于在未安装依赖时查看 --help
    from ultralytics import YOLO

    model = YOLO(str(args.model))

    predict_kwargs = {
        "source": str(args.source),
        "conf": args.conf,
        "imgsz": args.imgsz,
        "save": True,
        "project": "outputs",
        "name": "image_predictions",
        "exist_ok": True,
    }
    if args.device is not None:
        predict_kwargs["device"] = args.device

    model.predict(**predict_kwargs)

    print(f"预测完成，结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
