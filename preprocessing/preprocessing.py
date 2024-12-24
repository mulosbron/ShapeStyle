import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.filters import sobel
from skimage.feature import canny
from scipy import ndimage
import random
import tensorflow as tf
from tensorflow.keras import utils
import pandas as pd
import seaborn as sns
from joblib import dump  # Joblib'den dump fonksiyonunu doğrudan içe aktarın
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from mtcnn import MTCNN  # MTCNN kütüphanesini içe aktarın

# Pillow'un kesilmiş (truncated) görüntüleri yüklemesine izin verme (Opsiyonel)
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Uyarıları bastırmak (isteğe bağlı)
import warnings
warnings.filterwarnings('ignore')

# Yüz tespiti için MTCNN detektörünü başlatma
detector = MTCNN()

def extract_face(img, target_size=(224, 224)):
    '''
    Bu fonksiyon, farklı görüntülerden yüzleri tespit eder ve
    1) Yüzün sınırlayıcı kutusunu bulur
    2) Yüzün üst ve alt sınırlarını biraz genişletir
    3) Kare şeklinde kırpar
    4) Modelleme için hedef boyuta yeniden boyutlandırır
    '''
    # 1. Görüntüde yüzleri tespit etme
    results = detector.detect_faces(img)
    if not results:
        # Yüz bulunamazsa, görüntüyü oranı koruyarak kırp ve yeniden boyutlandır
        new_face = crop_and_resize(img, target_w=224, target_h=224)
    else:
        # İlk tespit edilen yüzü al
        x1, y1, width, height = results[0]['box']
        x2, y2 = x1 + width, y1 + height
        face = img[y1:y2, x1:x2]

        # 2. Yüzün üst ve alt sınırlarını 10 piksel genişletme
        adj_h = 10

        new_y1 = max(y1 - adj_h, 0)
        new_y2 = min(y1 + height + adj_h, img.shape[0])
        new_height = new_y2 - new_y1

        # 3. Kare şeklinde kırpma
        adj_w = int((new_height - width) / 2)

        new_x1 = max(x1 - adj_w, 0)
        new_x2 = min(x2 + adj_w, img.shape[1])
        new_face = img[new_y1:new_y2, new_x1:new_x2]

    # 4. Görüntüyü hedef boyuta yeniden boyutlandırma
    sqr_img = cv2.resize(new_face, target_size)
    return sqr_img

def crop_and_resize(image, target_w=224, target_h=224):
    '''
    Bu fonksiyon, görüntüyü hedef boyuta oranını koruyarak kırpar ve yeniden boyutlandırır
    '''
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
        crop_top = int((resize_h - target_h) / 4)  # Üst kısmı biraz kırp
        crop_bottom = crop_top + target_h
        new_img = img[crop_top:crop_bottom, :]
    else:
        new_img = cv2.resize(image, (target_w, target_h))

    return new_img

def cvt_gabor(image):
    '''
    Gabor filtresi uygulama fonksiyonu
    '''
    # Gabor filtresi parametreleri
    gabor_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    filtered_img = cv2.filter2D(image, cv2.CV_8UC3, gabor_kernel)
    return filtered_img

def create_data_files(directory, array, type=None):
    '''
    Bu fonksiyon, verilen dizindeki görüntüleri okur ve eğitim & test veri setleri oluşturur
    '''
    i = 0
    for category in categories:
        path = os.path.join(directory, category)  # Görüntü dizinine yol
        class_num = categories.index(category)  # Kategorilere sayı atama
        if not os.path.exists(path):
            print(f"Uyarı: Kategori klasörü bulunamadı: {path}")
            continue
        img_list = os.listdir(path)

        if type == 'aspect':
            try:
                for img in img_list:
                    img_path = os.path.join(path, img)
                    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Grayscale olarak oku
                    if img_array is None:
                        print(f"Uyarı: Görüntü okunamadı: {img_path}")
                        continue
                    img_array = crop_and_resize(img_array)  # Crop & resize
                    array.append([img_array, class_num])
                    i += 1
            except Exception as e:
                print(f'Error: \n category: {category}\n image: {img}\n {e}')

        elif type == 'gray':
            try:
                for img in img_list:
                    img_path = os.path.join(path, img)
                    img_array = cv2.imread(img_path)
                    if img_array is None:
                        print(f"Uyarı: Görüntü okunamadı: {img_path}")
                        continue
                    face_array = extract_face(img_array)  # Yüzü çıkar
                    gray_array = cv2.cvtColor(face_array, cv2.COLOR_BGR2GRAY)  # Gray'e çevir
                    gray_array = crop_and_resize(gray_array)  # Crop & resize
                    array.append([gray_array, class_num])  # [image, class]
                    i += 1
            except Exception as e:
                print(f'Error: \n category: {category}\n image: {img}\n {e}')

        elif type == 'rgb':
            try:
                for img in img_list:
                    img_path = os.path.join(path, img)
                    img_array = cv2.imread(img_path)
                    if img_array is None:
                        print(f"Uyarı: Görüntü okunamadı: {img_path}")
                        continue
                    face_array = extract_face(img_array)  # Yüzü çıkar
                    rgb_array = cv2.cvtColor(face_array, cv2.COLOR_BGR2RGB)  # RGB'ye çevir
                    rgb_array = crop_and_resize(rgb_array)  # Crop & resize
                    array.append([rgb_array, class_num])  # [image, class]
                    i += 1
            except Exception as e:
                print(f'Error: \n category: {category}\n image: {img}\n {e}')

        else:
            print("Lütfen görüntü tipi belirtin ['aspect', 'gray', 'rgb']")
            break

        # Her 200 görüntüde bir ilerlemeyi yazdır
        total_images = len(img_list) * len(categories)
        if i % 200 == 0 and i > 0:
            print(f"İşlenen görüntüler: {i} / {total_images}")

def show_img(num, img_array, title, ncols=1):
    '''
    Bu fonksiyon, bir görüntü dizisinden görüntüleri gösterir
    '''
    nrows = int(np.ceil(num / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    fig.suptitle(title, size=20)
    if nrows == 1 and ncols == 1:
        ax = [ax]
    elif nrows == 1 or ncols == 1:
        ax = ax.flatten()
    else:
        ax = ax.ravel()
    for i in range(num):
        img = img_array[i][0]
        if img.ndim == 2:
            ax[i].imshow(img, cmap='gray')
        else:
            ax[i].imshow(img)
        ax[i].set_title(label_map.get(img_array[i][1]), size=16)
        ax[i].axis('off')
    plt.tight_layout()
    plt.show()

def print_summary(X_train, X_test, y_train, y_test):
    '''
    Veri setlerinin özet bilgilerini yazdırır
    '''
    print(f'\nEğitim Veri Seti:\n')
    print(f'Özellik Şekli: {X_train.shape}')
    print(f'Etiket Şekli: {y_train.shape}')
    print(f'Sınıflar: {np.unique(np.argmax(y_train, axis=1))}')
    print(f'Her sınıfta görüntü sayısı: {np.bincount(np.argmax(y_train, axis=1))}')
    print(f'Maksimum Piksel Değeri: {np.amax(X_train)}')
    print('\n--------------------------------------\n')
    print(f'\nTest Veri Seti:\n')
    print(f'Özellik Şekli: {X_test.shape}')
    print(f'Etiket Şekli: {y_test.shape}')
    print(f'Sınıflar: {np.unique(np.argmax(y_test, axis=1))}')
    print(f'Her sınıfta görüntü sayısı: {np.bincount(np.argmax(y_test, axis=1))}')
    print(f'Maksimum Piksel Değeri: {np.amax(X_test)}')
    print('\n--------------------------------------\n')

def train_test_prep(training_data_array, testing_data_array):
    '''
    Veri setlerini karıştırır ve X_train, X_test, y_train, y_test olarak ayırır
    '''
    # Görüntüleri karıştırarak sınıfları rastgele dağıtma
    random.shuffle(training_data_array)
    random.shuffle(testing_data_array)

    # X_train, X_test, y_train, y_test'e ayırma
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for features, label in training_data_array:
        X_train.append(features)
        y_train.append(label)

    for features, label in testing_data_array:
        X_test.append(features)
        y_test.append(label)

    # Veri tipini değiştirme ve normalize etme
    X_train = np.array(X_train, dtype=float)
    X_test = np.array(X_test, dtype=float)

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Etiketleri kategorik yapma
    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)

    return (X_train, X_test, y_train, y_test)

def joblib_out(X_train, X_test, y_train, y_test, version, data_path):
    '''
    Modelleme için verileri joblib olarak kaydeder
    '''
    dump(X_train, os.path.join(data_path, f'X_train_{version}.joblib'), compress=True)
    dump(y_train, os.path.join(data_path, f'y_train_{version}.joblib'), compress=True)
    dump(X_test, os.path.join(data_path, f'X_test_{version}.joblib'), compress=True)
    dump(y_test, os.path.join(data_path, f'y_test_{version}.joblib'), compress=True)

def display_preprocessed_images():
    '''
    Eğitim setinizden her kategoriden birkaç örnek görüntü seçip işleyerek gösterir
    '''
    samples_per_category = 2  # Her kategoriden kaç örnek görüntü göstermek istediğinizi belirleyin
    examples = []

    for category in categories:
        category_path = os.path.join(train_dir, category)
        if not os.path.exists(category_path):
            print(f"Uyarı: Kategori klasörü bulunamadı: {category_path}")
            continue
        img_files = os.listdir(category_path)[:samples_per_category]  # İlk birkaç görüntüyü al
        for img_file in img_files:
            img_path = os.path.join(category_path, img_file)
            examples.append(img_path)

    if not examples:
        print("Uyarı: Hiçbir örnek görüntü bulunamadı.")
        return

    n_images = len(examples)
    fig, ax = plt.subplots(nrows=7, ncols=n_images, figsize=(n_images * 2.5, 7 * 2.5))
    plt.gray()
    ax = ax.ravel()

    for i, file in enumerate(examples):
        img = cv2.imread(file)
        if img is None:
            print(f"Uyarı: Görüntü okunamadı: {file}")
            continue
        new_img = extract_face(img)  # Yüzü çıkar

        # Auto resize by 224 - distort edilmiş
        rsz_img = cv2.resize(img, (224, 224))
        rsz_img = cv2.cvtColor(rsz_img, cv2.COLOR_BGR2GRAY)
        ax[i].imshow(rsz_img)
        ax[i].axis('off')

        # Aspect Ratio koruyarak kırpma ve yeniden boyutlandırma
        aspct_img = crop_and_resize(img, target_w=224, target_h=224)
        aspct_img = cv2.cvtColor(aspct_img, cv2.COLOR_BGR2GRAY)
        ax[i + n_images].imshow(aspct_img)
        ax[i + n_images].axis('off')

        # RGB görüntü
        rgb_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
        ax[i + 2 * n_images].imshow(rgb_img)
        ax[i + 2 * n_images].axis('off')

        # Grayscale görüntü
        gray_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        ax[i + 3 * n_images].imshow(gray_img)
        ax[i + 3 * n_images].axis('off')

        # Gabor filtresi uygulanmış görüntü
        gabor_img = cvt_gabor(gray_img)
        ax[i + 4 * n_images].imshow(gabor_img)
        ax[i + 4 * n_images].axis('off')

        # Sobel filtresi uygulanmış görüntü
        sobel_img = sobel(gray_img)
        ax[i + 5 * n_images].imshow(sobel_img)
        ax[i + 5 * n_images].axis('off')

        # Canny kenar algılama uygulanmış görüntü
        canny_img = canny(gray_img, sigma=1.5)
        ax[i + 6 * n_images].imshow(canny_img)
        ax[i + 6 * n_images].axis('off')

    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    # Başlıkları ekleme
    plt.figtext(x=0.105, y=0.8, s="Auto Resize", fontsize=15, rotation=90)
    plt.figtext(x=0.105, y=0.68, s="w/Aspect Ratio", fontsize=15, rotation=90)
    plt.figtext(x=0.105, y=0.6, s="BBox", fontsize=15, rotation=90)
    plt.figtext(x=0.105, y=0.48, s="BBox-RGB", fontsize=15, rotation=90)
    plt.figtext(x=0.105, y=0.37, s="BBox-Gabor", fontsize=15, rotation=90)
    plt.figtext(x=0.105, y=0.26, s="BBox-Sobel", fontsize=15, rotation=90)
    plt.figtext(x=0.105, y=0.13, s="BBox-Canny Edges", fontsize=15, rotation=90)
    plt.figtext(x=0.16, y=0.89, s="Heart", fontsize=15)
    plt.figtext(x=0.26, y=0.89, s="Oblong", fontsize=15)
    plt.figtext(x=0.38, y=0.89, s="Oval", fontsize=15)
    plt.figtext(x=0.49, y=0.89, s="Round", fontsize=15)
    plt.figtext(x=0.6, y=0.89, s="Square", fontsize=15)
    plt.figtext(x=0.68, y=0.89, s="|", fontsize=15)
    plt.figtext(x=0.7, y=0.89, s="Long Portrait", fontsize=15)
    plt.figtext(x=0.8, y=0.89, s="Wide Landscape", fontsize=15)
    plt.show()

# --- Başlangıç Ayarları ---

# Doğru yol
base_dir = '/FaceShape Dataset'  # Doğru yol
train_dir = os.path.join(base_dir, 'training_set')
test_dir = os.path.join(base_dir, 'testing_set')
categories = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
label_map = {0: 'Heart', 1: 'Oblong', 2: 'Oval', 3: 'Round', 4: 'Square'}
num_classes = len(categories)  # num_classes değişkenini tanımlayın

# Yol kontrolleri
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Eğitim dizini bulunamadı: {train_dir}")

if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Test dizini bulunamadı: {test_dir}")

# Görüntüleri gösterme (isteğe bağlı)
# Eğer örnek görüntüleri görmek istemiyorsanız, aşağıdaki döngüyü kaldırabilirsiniz
# Ancak, burada `display_preprocessed_images` fonksiyonunu kullanacağız

# --- Veri İşleme ve Joblib ile Kaydetme ---

# Veri oluşturma
training_data_apr = []
testing_data_apr = []

print("Aspect Ratio Koruyarak Veri İşleniyor...")
create_data_files(train_dir, training_data_apr, 'aspect')
create_data_files(test_dir, testing_data_apr, 'aspect')

# Veri setlerinin özetini yazdırma
X_train_apr, X_test_apr, y_train_apr, y_test_apr = train_test_prep(training_data_apr, testing_data_apr)
print_summary(X_train_apr, X_test_apr, y_train_apr, y_test_apr)

# Sınıf dağılımını kontrol etme
counter = Counter([label for _, label in training_data_apr])
print(f"Eğitim veri setindeki sınıf dağılımı: {counter}")
counter = Counter([label for _, label in testing_data_apr])
print(f"Test veri setindeki sınıf dağılımı: {counter}")

# Veri setlerini joblib olarak kaydetme
data_path = os.path.join(base_dir, 'pickles')  # Doğru yol
if not os.path.exists(data_path):
    os.makedirs(data_path)

joblib_out(X_train_apr, X_test_apr, y_train_apr, y_test_apr, version='apr', data_path=data_path)

print("Aspect Ratio ile oluşturulan joblib dosyaları başarıyla oluşturuldu.")

# Version GRAY: Detect the face with bounding box - Grayscale

training_data_gray = []
testing_data_gray = []

print("Grayscale ile Veri İşleniyor...")
create_data_files(train_dir, training_data_gray, 'gray')
create_data_files(test_dir, testing_data_gray, 'gray')

print(f'Training Images: {len(training_data_gray)}')
print(f'Testing Images: {len(testing_data_gray)}')

X_train_gray, X_test_gray, y_train_gray, y_test_gray = train_test_prep(training_data_gray, testing_data_gray)

print_summary(X_train_gray, X_test_gray, y_train_gray, y_test_gray)

# Sınıf dağılımını kontrol etme
counter = Counter([label for _, label in training_data_gray])
print(f"Eğitim veri setindeki sınıf dağılımı: {counter}")
counter = Counter([label for _, label in testing_data_gray])
print(f"Test veri setindeki sınıf dağılımı: {counter}")

# Veri setlerini joblib olarak kaydetme
joblib_out(X_train_gray, X_test_gray, y_train_gray, y_test_gray, version='gray', data_path=data_path)

print("Grayscale ile oluşturulan joblib dosyaları başarıyla oluşturuldu.")

# Version RGB: Detect the face with bounding box - in RGB color

training_data_rgb = []
testing_data_rgb = []

print("RGB ile Veri İşleniyor...")
create_data_files(train_dir, training_data_rgb, 'rgb')
create_data_files(test_dir, testing_data_rgb, 'rgb')

print(f'Training Images: {len(training_data_rgb)}')
print(f'Testing Images: {len(testing_data_rgb)}')

X_train_rgb, X_test_rgb, y_train_rgb, y_test_rgb = train_test_prep(training_data_rgb, testing_data_rgb)

print_summary(X_train_rgb, X_test_rgb, y_train_rgb, y_test_rgb)

# Sınıf dağılımını kontrol etme
counter = Counter([label for _, label in training_data_rgb])
print(f"Eğitim veri setindeki sınıf dağılımı: {counter}")
counter = Counter([label for _, label in testing_data_rgb])
print(f"Test veri setindeki sınıf dağılımı: {counter}")

# Veri setlerini joblib olarak kaydetme
joblib_out(X_train_rgb, X_test_rgb, y_train_rgb, y_test_rgb, version='rgb', data_path=data_path)

print("RGB ile oluşturulan joblib dosyaları başarıyla oluşturuldu.")

# Örnek görüntüleri gösterme (isteğe bağlı)
print("Örnek görüntüler işleniyor ve gösteriliyor...")
display_preprocessed_images()

print("Tüm veriler başarıyla işlendi ve joblib dosyaları oluşturuldu.")
