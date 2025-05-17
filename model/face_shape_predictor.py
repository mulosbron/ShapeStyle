import os
import numpy as np
import cv2
from mtcnn import MTCNN
from keras.models import load_model
import matplotlib.pyplot as plt


def extract_face(img, detector, target_size=(224, 224)):
    results = detector.detect_faces(img)
    if not results:
        new_face = crop_and_resize(img, target_w=224, target_h=224)
    else:
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


def crop_and_resize(image, target_w=224, target_h=224):
    if image.ndim == 2:
        img_h, img_w = image.shape
    elif image.ndim == 3:
        img_h, img_w, channels = image.shape
    target_aspect_ratio = target_w / target_h
    input_aspect_ratio = img_w / img_h
    if input_aspect_ratio > target_aspect_ratio:
        resize_w = int(input_aspect_ratio * target_h)
        resize_h = target_h
        img = cv2.resize(image, (resize_w, resize_h))
        crop_left = int((resize_w - target_w) / 2)
        crop_right = crop_left + target_w
        new_img = img[:, crop_left:crop_right]
    elif input_aspect_ratio < target_aspect_ratio:
        resize_w = target_w
        resize_h = int(target_w / input_aspect_ratio)
        img = cv2.resize(image, (resize_w, resize_h))
        crop_top = int((resize_h - target_h) / 4)
        crop_bottom = crop_top + target_h
        new_img = img[crop_top:crop_bottom, :]
    else:
        new_img = cv2.resize(image, (target_w, target_h))
    return new_img


def predict_face_shape(image_path, model, detector, label_map):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image could not be loaded: {image_path}")
    processed_img = extract_face(img, detector)
    rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    normalized_img = rgb_img.astype('float32') / 255.0
    input_img = np.expand_dims(normalized_img, axis=0)
    predictions = model.predict(input_img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = label_map.get(predicted_class, "Bilinmiyor")
    confidence = predictions[0][predicted_class]
    return predicted_label, confidence, predictions[0], rgb_img


def plot_and_save(image, label, conf, all_confidences, label_map, save_path):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Predicted Face Shape: {label} ({conf * 100:.2f}% Confidence)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    classes = list(label_map.values())
    confidences = all_confidences * 100
    bar_colors = ['gray'] * len(classes)
    predicted_index = list(label_map.keys())[list(label_map.values()).index(label)]
    bar_colors[predicted_index] = 'blue'
    plt.bar(classes, confidences, color=bar_colors)
    plt.xlabel('Face Shape Classes')
    plt.ylabel('Confidence (%)')
    plt.title('Face Shape Predictions')
    plt.ylim(0, 105)

    for i, v in enumerate(confidences):
        plt.text(i, v + 1, f"{v:.2f}%", ha='center')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    input_folder = "../input_imgs"
    output_folder = "../output_results"
    model_path = "face_shape_model_vgg16_rgb.h5"
    label_map = {0: 'Heart', 1: 'Oblong', 2: 'Oval', 3: 'Round', 4: 'Square'}

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    detector = MTCNN()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
    model = load_model(model_path)

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(supported_extensions):
            continue
        image_path = os.path.join(input_folder, filename)
        try:
            label, conf, all_confidences, rgb_img = predict_face_shape(image_path, model, detector, label_map)
            print(f"{filename}: {label} ({conf * 100:.2f}% confidence)")
            for idx, confidence in enumerate(all_confidences):
                print(f"  {label_map[idx]}: {confidence * 100:.2f}%")
            output_filename = os.path.splitext(filename)[0] + "_prediction.png"
            output_plot_path = os.path.join(output_folder, output_filename)
            plot_and_save(rgb_img, label, conf, all_confidences, label_map, output_plot_path)
            print(f"  Prediction result image saved: {output_plot_path}\n")
        except Exception as e:
            print(f"{filename}: Error: {e}\n")
