"""Flask 人脸检测 Web 应用。"""

from __future__ import annotations

import uuid
from pathlib import Path

import cv2
from flask import Flask, flash, redirect, render_template, request, url_for
from ultralytics import YOLO


# 允许上传的文件类型
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# 定义关键目录路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = PROJECT_ROOT / "app" / "static" / "uploads"
RESULT_DIR = PROJECT_ROOT / "app" / "static" / "results"
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"


def create_app() -> Flask:
    """创建并配置 Flask 应用。"""
    app = Flask(__name__)
    app.secret_key = "face_detection_yolov8_demo_secret"

    # 启动时确保目录存在
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # 尝试加载模型，若失败则在页面给出提示
    model = None
    if MODEL_PATH.exists():
        model = YOLO(str(MODEL_PATH))
    else:
        print(f"[警告] 未找到模型文件: {MODEL_PATH}")

    def is_allowed(filename: str) -> bool:
        """检查文件扩展名是否允许。"""
        return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

    @app.route("/", methods=["GET", "POST"])
    def index():
        """首页：上传图片并执行检测。"""
        if request.method == "POST":
            if model is None:
                flash("尚未找到 models/best.pt，请先完成训练并放置模型文件。")
                return redirect(url_for("index"))

            if "image" not in request.files:
                flash("未检测到上传文件。")
                return redirect(url_for("index"))

            file = request.files["image"]
            if file.filename == "":
                flash("请选择图片文件。")
                return redirect(url_for("index"))

            if not is_allowed(file.filename):
                flash("仅支持 jpg/jpeg/png 格式。")
                return redirect(url_for("index"))

            suffix = Path(file.filename).suffix.lower()
            unique_name = f"{uuid.uuid4().hex}{suffix}"

            upload_path = UPLOAD_DIR / unique_name
            result_path = RESULT_DIR / unique_name

            # 保存上传文件
            file.save(upload_path)

            # 使用 YOLOv8 模型进行预测
            results = model.predict(source=str(upload_path), conf=0.25, verbose=False)
            result = results[0]

            # 读取原图，并绘制人脸框
            image = cv2.imread(str(upload_path))
            if image is None:
                flash("图片读取失败，请重试。")
                return redirect(url_for("index"))

            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
            scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else []

            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    image,
                    f"face {score:.2f}",
                    (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            # 保存检测结果图
            cv2.imwrite(str(result_path), image)

            return render_template(
                "index.html",
                original_image=url_for("static", filename=f"uploads/{unique_name}"),
                result_image=url_for("static", filename=f"results/{unique_name}"),
            )

        return render_template("index.html", original_image=None, result_image=None)

    return app


if __name__ == "__main__":
    # 直接运行本文件时，启动开发服务器
    flask_app = create_app()
    flask_app.run(host="0.0.0.0", port=5000, debug=True)
