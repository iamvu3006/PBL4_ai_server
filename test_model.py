import pickle
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image

# === 0️⃣ Cố định seed để kết quả ổn định ===
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

print("🔧 Đang chạy TensorFlow trên CPU...")
tf.config.set_visible_devices([], 'GPU')  # tắt GPU để tránh cảnh báo CUDA

# === 1️⃣ Load model pickle gốc ===
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

# === 5️⃣ Dự đoán ảnh ===
try:
    test_images = ["test1.jpg".split(", ")]
    for path in test_images:
        img = image.load_img(path, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array, verbose=0)
        label = "🍃 Lá tốt" if prediction[0][0] < 0.51 else "🍂 Lá xấu"
        print(f"✅ {path}: {label} ({prediction[0][0]:.4f})")
except Exception as e:
    print("❌ Lỗi khi dự đoán:", e)