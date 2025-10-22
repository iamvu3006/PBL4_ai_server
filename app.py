import os
import pickle
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template, jsonify

# === 0️⃣ Cấu hình chung ===
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

print("🔧 Đang chạy TensorFlow trên CPU...")
tf.config.set_visible_devices([], 'GPU')  # Tắt GPU để tránh cảnh báo CUDA

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === 1️⃣ Load model từ pickle ===
MODEL_PATH = "cnn_model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        old_model = pickle.load(f)
    print("✅ Đã load model .pkl thành công!")
except Exception as e:
    print("❌ Lỗi khi load model .pkl:", e)
    exit()

# === 2️⃣ Tạo lại đúng kiến trúc model ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

print("✅ Model đã khởi tạo lại thành công!")

# --- Flask app setup ---
app = Flask(_name_)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# === 3️⃣ Hàm xử lý dự đoán ===
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img = img.convert("RGB")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array, verbose=0)
    label = "🍃 Lá tốt" if prediction[0][0] < 0.51 else "🍂 Lá xấu"
    return label, float(prediction[0][0])

# === 4️⃣ Route trang chủ (web upload) ===
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return render_template("index.html", result="⚠️ Chưa chọn ảnh!")
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(save_path)

        label, prob = predict_image(save_path)
        return render_template("index.html", result=f"{label} ({prob:.4f})", img_path=save_path)
    return render_template("index.html")

# === 5️⃣ Route API (POST /api/predict) ===
@app.route("/api/predict", methods=["POST"])
def api_predict():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "Chưa upload file!"}), 400
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(save_path)
    label, prob = predict_image(save_path)
    return jsonify({"label": label, "probability": prob})

# === 6️⃣ Chạy server ===
if _name_ == "_main_":
    app.run(host="0.0.0.0", port=5000)