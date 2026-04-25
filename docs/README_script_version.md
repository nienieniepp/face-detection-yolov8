# 纯脚本版本说明（非 Flask）

本文档说明如何仅通过脚本完成人脸检测项目全流程：

1. 数据切分 + 生成 `data/face.yaml`
2. 训练 YOLOv8n
3. 评估模型指标
4. 图片预测
5. 视频预测

> 所有命令都需要在项目根目录执行。

---

## 1) 切分数据并生成配置

```bash
python scripts/01_split_dataset.py --clean
```

默认输入：

- `data/wider_face_yolo/images`
- `data/wider_face_yolo/labels`

默认输出：

- `data/wider_face_yolo_split/images/{train,val,test}`
- `data/wider_face_yolo_split/labels/{train,val,test}`
- `data/face.yaml`

---

## 2) 训练 YOLOv8n 并导出最佳模型

```bash
python scripts/02_train_yolov8.py
```

默认行为：

- 使用 `data/face.yaml`
- 加载 `yolov8n.pt`
- 训练结果写入 `runs/face_yolov8n`
- 自动复制最佳模型到 `models/best.pt`

---

## 3) 评估模型

```bash
python scripts/03_evaluate_model.py
```

终端将打印：

- Precision
- Recall
- mAP50
- mAP50-95

---

## 4) 图片预测

单张图：

```bash
python scripts/04_predict_image.py --source test_images/sample.jpg
```

目录：

```bash
python scripts/04_predict_image.py --source test_images/
```

输出目录：

- `outputs/image_predictions/`

---

## 5) 视频预测

```bash
python scripts/05_predict_video.py --source test_videos/sample.mp4
```

输出目录：

- `outputs/video_predictions/`

---

## 常见问题

1. **提示找不到 `models/best.pt`**：
   - 先运行训练脚本 `python scripts/02_train_yolov8.py`。

2. **显存不足（CUDA OOM）**：
   - 在训练脚本中降低 `--batch`，如 `--batch 8`。

3. **CPU 推理较慢**：
   - 可在命令中指定 `--device 0` 使用 GPU（前提是环境可用）。
