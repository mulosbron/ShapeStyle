import numpy as np
import cv2
from mtcnn import MTCNN
from keras.models import load_model
from flask import Flask, request, jsonify
from flask_cors import CORS
import matplotlib.pyplot as plt
import base64
import tensorflow as tf

app = Flask(__name__)
CORS(app)  # Tüm rotalar için CORS'u etkinleştirir

# Klasörlerin Tanımlanması
backend_dir = r"C:\Users\duggy\OneDrive\Belgeler\Github\AIProject\sac_onerisi\backend"
input_folder = rf"{backend_dir}\input"
output_folder = rf"{backend_dir}\output"
model_path = r"C:\Users\duggy\OneDrive\Belgeler\Github\AIProject\model\face_shape_model_vgg16_rgb.h5"

# Etiket Haritası
label_map = {0: 'Heart', 1: 'Oblong', 2: 'Oval', 3: 'Round', 4: 'Square'}

# TensorFlow Graph ve Session Yönetimi
graph = tf.get_default_graph()
session = tf.Session()
with graph.as_default():
    tf.keras.backend.set_session(session)
    model = load_model(model_path)
    detector = MTCNN()

def crop_and_resize(image, target_w=224, target_h=224):
    if image.ndim == 2:
        img_h, img_w = image.shape
    elif image.ndim == 3:
        img_h, img_w, channels = image.shape
    else:
        raise ValueError("Unsupported image dimensions")

    target_aspect_ratio = target_w / target_h
    input_aspect_ratio = img_w / img_h

    if input_aspect_ratio > target_aspect_ratio:
        resize_w = int(input_aspect_ratio * target_h)
        resize_h = target_h
        img = cv2.resize(image, (resize_w, resize_h))
        crop_left = int((resize_w - target_w) / 2)
        new_img = img[:, crop_left:crop_left + target_w]
    elif input_aspect_ratio < target_aspect_ratio:
        resize_w = target_w
        resize_h = int(target_w / input_aspect_ratio)
        img = cv2.resize(image, (resize_w, resize_h))
        crop_top = int((resize_h - target_h) / 4)
        new_img = img[crop_top:crop_top + target_h, :]
    else:
        new_img = cv2.resize(image, (target_w, target_h))

    return new_img

def extract_face(img, detector, target_size=(224, 224)):
    with graph.as_default():
        tf.keras.backend.set_session(session)
        results = detector.detect_faces(img)
        if not results:
            # Yüz bulunamadı, görseli kırpar ve yeniden boyutlandırır
            new_face = crop_and_resize(img, target_w=224, target_h=224)
        else:
            # Yüz bulundu, koordinatları kullanarak kırpar ve genişletir
            x1, y1, width, height = results[0]['box']
            x2, y2 = x1 + width, y1 + height
            adj_h = 10
            new_y1 = max(y1 - adj_h, 0)
            new_y2 = min(y1 + height + adj_h, img.shape[0])
            new_height = new_y2 - new_y1
            adj_w = int((new_height - width) / 2)
            new_x1 = max(x1 - adj_w, 0)
            new_x2 = min(x2 + adj_w, img.shape[1])
            new_face = img[new_y1:new_y2, new_x1:new_x2]

        sqr_img = cv2.resize(new_face, target_size)
        return sqr_img

def predict_and_visualize(image_path, output_image_path):
    with graph.as_default():
        tf.keras.backend.set_session(session)
        img = cv2.imread(image_path)
        if img is None:
            return False  # Görsel yüklenemedi

        processed_img = extract_face(img, detector)
        rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        normalized_img = rgb_img.astype('float32') / 255.0
        input_img = np.expand_dims(normalized_img, axis=0)

        predictions = model.predict(input_img)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = label_map.get(predicted_class, "Bilinmiyor")
        confidence = predictions[0][predicted_class]
        classes = list(label_map.values())
        confidences = predictions[0]

        # Tahmin Sonuçlarını Görselleştirme
        plt.figure(figsize=(10, 5))

        # Yüklenen Görsel ve Tahmin
        plt.subplot(1, 2, 1)
        plt.imshow(rgb_img)
        plt.title(f"Tahmin Edilen Yüz Şekli: {predicted_label} ({confidence * 100:.2f}% Doğruluk)")
        plt.axis('off')

        # Güven Bar Grafiği
        plt.subplot(1, 2, 2)
        bar_colors = ['gray'] * len(classes)
        bar_colors[predicted_class] = 'blue'
        plt.bar(classes, confidences * 100, color=bar_colors)
        plt.xlabel('Yüz Şekli Sınıfları')
        plt.ylabel('Doğruluk (%)')
        plt.title('Yüz Şekli Tahminleri')
        plt.ylim(0, 105)

        for i, v in enumerate(confidences * 100):
            plt.text(i, v + 1, f"{v:.2f}%", ha='center')

        plt.tight_layout()
        plt.savefig(output_image_path)
        plt.close()
        return True

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "Dosya bulunamadı"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Hiçbir dosya seçilmedi"}), 400
    filename = file.filename

    # Dosya adı ve uzantısını ayrıştırma (os.path.splitext yerine)
    if '.' in filename:
        name, ext = filename.rsplit('.', 1)
        ext = '.' + ext.lower()
    else:
        name = filename
        ext = ''

    # Desteklenen dosya türlerini kontrol etme
    if ext not in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
        return jsonify({"error": "Desteklenmeyen dosya türü"}), 400

    # Dosya adını ve yollarını oluşturma
    input_filename = f"{name}_input{ext}"
    input_path = f"{input_folder}\\{input_filename}"
    output_filename = f"{name}_output.png"
    output_path = f"{output_folder}\\{output_filename}"

    # Dosyayı kaydetme
    try:
        file.save(input_path)
    except Exception as e:
        return jsonify({"error": f"Dosya kaydedilirken hata oluştu: {str(e)}"}), 500

    # Görseli işleme
    try:
        success = predict_and_visualize(input_path, output_path)
        if not success:
            return jsonify({"error": "Görsel işlenemedi"}), 500
    except Exception as e:
        return jsonify({"error": f"Görsel işlenirken hata oluştu: {str(e)}"}), 500

    # İşlenen görseli base64 formatına dönüştürme
    try:
        with open(output_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        return jsonify({"error": f"İşlenen görsel okunurken hata oluştu: {str(e)}"}), 500

    return jsonify({"output_image": encoded_string}), 200

if __name__ == '__main__':
    app.run(debug=True)