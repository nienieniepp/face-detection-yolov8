"""将 WIDER FACE YOLO 数据集按比例切分为 train/val/test。"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


# 支持的图片后缀（统一小写比较）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="切分 WIDER FACE YOLO 数据集")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/wider_face_yolo"),
        help="原始数据集目录，需包含 images/ 与 labels/",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("data/wider_face_yolo_split"),
        help="切分后输出目录",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，保证可复现")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="默认使用硬链接；若希望复制文件请开启该参数",
    )
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    """校验切分比例是否合理。"""
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"切分比例之和必须为 1.0，当前为 {ratio_sum:.4f}")


def collect_image_label_pairs(source_dir: Path) -> list[tuple[Path, Path]]:
    """收集并匹配图片与标签文件。"""
    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError("source 目录下必须包含 images/ 和 labels/ 子目录")

    pairs: list[tuple[Path, Path]] = []
    missing_labels: list[Path] = []

    # 遍历所有图片，按相同 stem 匹配标签 txt
    for image_path in sorted(images_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        relative = image_path.relative_to(images_dir)
        label_path = (labels_dir / relative).with_suffix(".txt")
        if label_path.exists():
            pairs.append((image_path, label_path))
        else:
            missing_labels.append(image_path)

    if missing_labels:
        print(f"[警告] 有 {len(missing_labels)} 张图片未找到对应标签，将被跳过。")

    if not pairs:
        raise RuntimeError("未找到可用的图片-标签配对，请检查数据目录。")

    return pairs


def prepare_output_dirs(target_dir: Path) -> None:
    """创建输出目录结构。"""
    for split_name in ("train", "val", "test"):
        (target_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (target_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)


def transfer_file(src: Path, dst: Path, copy_files: bool) -> None:
    """传输文件：可选择硬链接或复制。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()

    if copy_files:
        shutil.copy2(src, dst)
    else:
        try:
            dst.hardlink_to(src)
        except OSError:
            # 如果硬链接失败（例如跨分区），自动退化为复制
            shutil.copy2(src, dst)


def split_pairs(
    pairs: list[tuple[Path, Path]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[tuple[Path, Path]]]:
    """按比例切分图片标签对。"""
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


def materialize_splits(
    split_map: dict[str, list[tuple[Path, Path]]],
    source_dir: Path,
    target_dir: Path,
    copy_files: bool,
) -> None:
    """将切分结果写入目标目录。"""
    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels"

    for split_name, items in split_map.items():
        for image_path, label_path in items:
            image_rel = image_path.relative_to(images_dir)
            label_rel = label_path.relative_to(labels_dir)

            target_image = target_dir / "images" / split_name / image_rel
            target_label = target_dir / "labels" / split_name / label_rel

            transfer_file(image_path, target_image, copy_files)
            transfer_file(label_path, target_label, copy_files)


def main() -> None:
    """程序入口。"""
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    pairs = collect_image_label_pairs(args.source)
    prepare_output_dirs(args.target)
    split_map = split_pairs(pairs, args.train_ratio, args.val_ratio, args.seed)
    materialize_splits(split_map, args.source, args.target, args.copy)

    print("切分完成：")
    for split_name, items in split_map.items():
        print(f"  {split_name}: {len(items)}")
    print(f"输出目录：{args.target}")


if __name__ == "__main__":
    main()
