# face-detection-yolov8

一个面向深度学习初学者的完整课程项目：使用 **YOLOv8** 在 **WIDER FACE（YOLO 标注格式）** 数据集上进行人脸检测训练，并通过 Flask 提供网页推理演示。

---

## 1. 项目简介

本项目目标是帮助你从 0 到 1 完成人脸检测项目闭环：

- 数据准备与自动切分（train/val/test）。
- 使用 YOLOv8n 进行训练与验证。
- 在测试图片上进行检测可视化。
- 将最佳模型接入 Flask Web 应用，实现网页上传检测。

---

## 2. 数据集说明

你已在本地准备好以下目录：

```text
data/wider_face_yolo/
├── images/
└── labels/
```

说明：

- `images/` 中每张图片都应有对应的 YOLO 标签文件（`.txt`）。
- 标签路径与图片路径（相对目录）应一致，仅后缀不同。

项目提供了自动切分脚本，会将数据切分为：

```text
data/wider_face_yolo_split/
├── images/train | val | test
└── labels/train | val | test
```

切分比例：`train=80%`，`val=10%`，`test=10%`。

---

## 3. 项目结构

```text
face-detection-yolov8/
├── app/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── uploads/
│       └── results/
├── data/
│   └── face.yaml
├── models/
├── notebooks/
│   └── 01_train_yolov8_face_detection.ipynb
├── scripts/
│   └── split_dataset.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 4. 安装步骤

> 建议使用 Python 3.10+。

1) 创建并激活虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2) 安装依赖：

```bash
pip install -r requirements.txt
```

---

## 5. 训练步骤（命令行方式）

### 第一步：切分数据集

```bash
python scripts/split_dataset.py \
  --source data/wider_face_yolo \
  --target data/wider_face_yolo_split \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42 \
  --copy
```

### 第二步：开始训练

你可以使用 Notebook（推荐初学者），或在 Python 脚本中调用 Ultralytics API。

Notebook 文件：

```text
notebooks/01_train_yolov8_face_detection.ipynb
```

Notebook 中已包含以下流程：

- 检查数据路径
- 运行数据切分
- 加载 `yolov8n.pt`
- 训练与验证
- 测试与可视化
- 导出最佳模型到 `models/best.pt`

---

## 6. Flask Web 应用运行步骤

确保你已经得到训练后的模型：

```text
models/best.pt
```

然后运行：

```bash
python app/app.py
```

打开浏览器访问：

```text
http://127.0.0.1:5000
```

功能说明：

- 支持上传 `jpg/jpeg/png` 图片。
- 自动进行人脸检测并绘制框。
- 页面同时展示原图与检测结果图。

---

## 7. GitHub 上传说明

由于数据和权重通常较大，本项目默认忽略以下内容：

- `data/`（保留 `data/face.yaml`）
- `runs/`
- `*.pt`
- `app/static/uploads/`
- `app/static/results/`

建议流程：

```bash
git init
git add .
git commit -m "feat: initialize face detection yolov8 project"
git branch -M main
git remote add origin <你的仓库地址>
git push -u origin main
```

---

## 8. 团队分工表（示例）

| 成员 | 角色 | 负责内容 |
|---|---|---|
| 张三 | 数据工程 | 数据清洗、YOLO 标签检查、数据切分 |
| 李四 | 模型工程 | YOLOv8 训练、调参、结果评估 |
| 王五 | 后端工程 | Flask 接口与推理逻辑实现 |
| 赵六 | 前端工程 | 上传页面与结果展示优化 |
| 全体 | 协作与文档 | README 撰写、代码 review、演示汇报 |

---

## 9. 初学者建议

- 先跑通最小流程，再逐步调参（如 `epochs`、`imgsz`、`batch`）。
- 如果显存不足，先减小 `batch` 或 `imgsz`。
- 每次训练记录指标与配置，便于复现与比较。

祝你训练顺利！
