"""纯脚本版本：切分数据集并生成 YOLOv8 配置文件。"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


# 中文注释：支持的图片后缀
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    """中文注释：解析命令行参数。"""
    parser = argparse.ArgumentParser(description="切分 WIDER FACE YOLO 数据并生成 face.yaml")
    parser.add_argument("--source", type=Path, default=Path("data/wider_face_yolo"), help="原始数据目录")
    parser.add_argument("--target", type=Path, default=Path("data/wider_face_yolo_split"), help="切分输出目录")
    parser.add_argument("--yaml-path", type=Path, default=Path("data/face.yaml"), help="生成的 face.yaml 路径")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--clean", action="store_true", help="切分前清理目标目录")
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    """中文注释：确保比例之和为 1。"""
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"切分比例之和必须为 1.0，当前为 {ratio_sum:.4f}")


def collect_pairs(source_dir: Path) -> list[tuple[Path, Path]]:
    """中文注释：收集图片与标签配对。"""
    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            "数据目录不完整：请确保存在 data/wider_face_yolo/images 与 data/wider_face_yolo/labels"
        )

    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(images_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = image_path.relative_to(images_dir)
        label_path = (labels_dir / relative).with_suffix(".txt")
        if label_path.exists():
            pairs.append((image_path, label_path))

    if not pairs:
        raise RuntimeError("未找到有效的图片-标签配对，请检查数据集内容。")

    return pairs


def prepare_dirs(target_dir: Path, clean: bool) -> None:
    """中文注释：准备 train/val/test 目录结构。"""
    if clean and target_dir.exists():
        shutil.rmtree(target_dir)

    for split in ("train", "val", "test"):
        (target_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (target_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_item(src: Path, dst: Path) -> None:
    """中文注释：复制文件，并确保父目录存在。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def split_pairs(
    pairs: list[tuple[Path, Path]], train_ratio: float, val_ratio: float, seed: int
) -> dict[str, list[tuple[Path, Path]]]:
    """中文注释：按比例切分数据。"""
    random.seed(seed)
    random.shuffle(pairs)

    total = len(pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    return {
        "train": pairs[:train_end],
        "val": pairs[train_end:val_end],
        "test": pairs[val_end:],
    }


def write_split_data(
    split_map: dict[str, list[tuple[Path, Path]]], source_dir: Path, target_dir: Path
) -> None:
    """中文注释：将切分后的图片与标签写入目标目录。"""
    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels"

    for split, items in split_map.items():
        for image_path, label_path in items:
            image_rel = image_path.relative_to(images_dir)
            label_rel = label_path.relative_to(labels_dir)
            copy_item(image_path, target_dir / "images" / split / image_rel)
            copy_item(label_path, target_dir / "labels" / split / label_rel)


def generate_face_yaml(yaml_path: Path, split_root: Path) -> None:
    """中文注释：生成 YOLOv8 训练需要的 face.yaml 文件。"""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_content = (
        "# YOLOv8 数据集配置文件\n"
        f"path: {split_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: face\n"
    )
    yaml_path.write_text(yaml_content, encoding="utf-8")


def main() -> None:
    """中文注释：脚本主流程。"""
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    pairs = collect_pairs(args.source)
    prepare_dirs(args.target, clean=args.clean)
    split_map = split_pairs(pairs, args.train_ratio, args.val_ratio, args.seed)
    write_split_data(split_map, args.source, args.target)
    generate_face_yaml(args.yaml_path, args.target)

    print("数据切分完成：")
    for split, items in split_map.items():
        print(f"  {split}: {len(items)}")
    print(f"切分目录：{args.target}")
    print(f"配置文件：{args.yaml_path}")


if __name__ == "__main__":
    main()
