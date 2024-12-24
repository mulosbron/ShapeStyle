import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import random

# --- Ayarlar ---
base_dir = '/FaceShape Dataset'  # Doğru yol
data_path = os.path.join(base_dir, 'pickles')  # Joblib dosyalarının bulunduğu yol

categories = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
label_map = {0: 'Heart', 1: 'Oblong', 2: 'Oval', 3: 'Round', 4: 'Square'}

def load_data(version):
    '''
    Belirtilen versiyona ait joblib dosyalarını yükler.
    '''
    try:
        X_train = joblib.load(os.path.join(data_path, f'X_train_{version}.joblib'))
        y_train = joblib.load(os.path.join(data_path, f'y_train_{version}.joblib'))
        X_test = joblib.load(os.path.join(data_path, f'X_test_{version}.joblib'))
        y_test = joblib.load(os.path.join(data_path, f'y_test_{version}.joblib'))
        print(f"{version} verileri başarıyla yüklendi.")
        return X_train, y_train, X_test, y_test
    except FileNotFoundError as e:
        print(f"Dosya bulunamadı: {e}")
        return None, None, None, None

def display_random_images(X, y, num_images=5):
    '''
    Verilen X ve y veri setlerinden rastgele num_images kadar görüntü seçer ve gösterir.
    '''
    if X is None or y is None:
        print("Veri seti boş. Görüntü gösterilemiyor.")
        return

    total_images = X.shape[0]
    if total_images < num_images:
        print(f"Veri setinde sadece {total_images} görüntü var. {num_images} görüntü gösterilemiyor.")
        num_images = total_images

    # Rastgele indeksler seçme
    random_indices = random.sample(range(total_images), num_images)

    # Seçilen görüntüleri ve etiketlerini alma
    selected_images = X[random_indices]
    selected_labels = np.argmax(y[random_indices], axis=1)  # Kategorik etiketlerden sınıf numarasını alma

    # Görüntüleri gösterme
    plt.figure(figsize=(15, 3))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        img = selected_images[i]
        label = label_map.get(selected_labels[i], 'Unknown')

        # Görüntü tipi kontrolü
        if img.ndim == 2:
            plt.imshow(img, cmap='gray')
        elif img.ndim == 3:
            plt.imshow(img)
        else:
            print(f"Görüntü {i+1} boyutunda desteklenmiyor: {img.shape}")
            continue

        plt.title(label)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    '''
    Ana fonksiyon: Kullanıcının seçtiği versiyonu yükler ve rastgele 5 görüntüyü gösterir.
    '''
    # Kullanıcıdan versiyon seçmesini isteyin
    print("Hangi veri setinden görüntü göstermek istersiniz?")
    print("1. Aspect Ratio ('apr')")
    print("2. Grayscale ('gray')")
    print("3. RGB ('rgb')")

    choice = input("Seçiminizi yapın (1/2/3): ")

    if choice == '1':
        version = 'apr'
    elif choice == '2':
        version = 'gray'
    elif choice == '3':
        version = 'rgb'
    else:
        print("Geçersiz seçim. Program sonlandırılıyor.")
        return

    # Veriyi yükleme
    X_train, y_train, X_test, y_test = load_data(version)

    if X_train is None:
        print("Veri yüklenemedi. Program sonlandırılıyor.")
        return

    # Hem eğitim hem de test verilerini birleştirme
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    # Rastgele 5 görüntüyü gösterme
    display_random_images(X, y, num_images=5)

if __name__ == "__main__":
    main()
