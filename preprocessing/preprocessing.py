import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.filters import sobel
from skimage.feature import canny
import random
from tensorflow.keras import utils
from joblib import dump
from collections import Counter
from mtcnn import MTCNN
from PIL import ImageFile
import warnings

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')
detector = MTCNN()

def extract_face(img, target_size=(224, 224)):
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

def cvt_gabor(image):
    gabor_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    filtered_img = cv2.filter2D(image, cv2.CV_8UC3, gabor_kernel)
    return filtered_img

def create_data_files(directory, array, type=None):
    i = 0
    for category in categories:
        path = os.path.join(directory, category)
        class_num = categories.index(category)
        if not os.path.exists(path):
            print(f"Warning: Category folder not found: {path}")
            continue
        img_list = os.listdir(path)

        if type == 'aspect':
            try:
                for img in img_list:
                    img_path = os.path.join(path, img)
                    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img_array is None:
                        print(f"Warning: Image could not be read: {img_path}")
                        continue
                    img_array = crop_and_resize(img_array)
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
                        print(f"Warning: Image could not be read: {img_path}")
                        continue
                    face_array = extract_face(img_array)
                    gray_array = cv2.cvtColor(face_array, cv2.COLOR_BGR2GRAY)
                    gray_array = crop_and_resize(gray_array)
                    array.append([gray_array, class_num])
                    i += 1
            except Exception as e:
                print(f'Error: \n category: {category}\n image: {img}\n {e}')

        elif type == 'rgb':
            try:
                for img in img_list:
                    img_path = os.path.join(path, img)
                    img_array = cv2.imread(img_path)
                    if img_array is None:
                        print(f"Warning: Image could not be read: {img_path}")
                        continue
                    face_array = extract_face(img_array)
                    rgb_array = cv2.cvtColor(face_array, cv2.COLOR_BGR2RGB)
                    rgb_array = crop_and_resize(rgb_array)
                    array.append([rgb_array, class_num])
                    i += 1
            except Exception as e:
                print(f'Error: \n category: {category}\n image: {img}\n {e}')

        else:
            print("Please specify image type ['aspect', 'gray', 'rgb']")
            break

        total_images = len(img_list) * len(categories)
        if i % 200 == 0 and i > 0:
            print(f"Processed images: {i} / {total_images}")

def show_img(num, img_array, title, ncols=1):
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
    print(f'\nTraining Dataset:\n')
    print(f'Feature Shape: {X_train.shape}')
    print(f'Label Shape: {y_train.shape}')
    print(f'Classes: {np.unique(np.argmax(y_train, axis=1))}')
    print(f'Number of images per class: {np.bincount(np.argmax(y_train, axis=1))}')
    print(f'Maximum Pixel Value: {np.amax(X_train)}')
    print('\n--------------------------------------\n')
    print(f'\nTest Dataset:\n')
    print(f'Feature Shape: {X_test.shape}')
    print(f'Label Shape: {y_test.shape}')
    print(f'Classes: {np.unique(np.argmax(y_test, axis=1))}')
    print(f'Number of images per class: {np.bincount(np.argmax(y_test, axis=1))}')
    print(f'Maximum Pixel Value: {np.amax(X_test)}')
    print('\n--------------------------------------\n')

def train_test_prep(training_data_array, testing_data_array):
    random.shuffle(training_data_array)
    random.shuffle(testing_data_array)

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

    X_train = np.array(X_train, dtype=float)
    X_test = np.array(X_test, dtype=float)

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)

    return (X_train, X_test, y_train, y_test)

def joblib_out(X_train, X_test, y_train, y_test, version, data_path):
    dump(X_train, os.path.join(data_path, f'X_train_{version}.joblib'), compress=True)
    dump(y_train, os.path.join(data_path, f'y_train_{version}.joblib'), compress=True)
    dump(X_test, os.path.join(data_path, f'X_test_{version}.joblib'), compress=True)
    dump(y_test, os.path.join(data_path, f'y_test_{version}.joblib'), compress=True)

def display_preprocessed_images():
    samples_per_category = 2
    examples = []

    for category in categories:
        category_path = os.path.join(train_dir, category)
        if not os.path.exists(category_path):
            print(f"Warning: Category folder not found: {category_path}")
            continue
        img_files = os.listdir(category_path)[:samples_per_category]
        for img_file in img_files:
            img_path = os.path.join(category_path, img_file)
            examples.append(img_path)

    if not examples:
        print("Warning: No example images found.")
        return

    n_images = len(examples)
    fig, ax = plt.subplots(nrows=7, ncols=n_images, figsize=(n_images * 2.5, 7 * 2.5))
    plt.gray()
    ax = ax.ravel()

    for i, file in enumerate(examples):
        img = cv2.imread(file)
        if img is None:
            print(f"Warning: Image could not be read: {file}")
            continue
        new_img = extract_face(img)

        rsz_img = cv2.resize(img, (224, 224))
        rsz_img = cv2.cvtColor(rsz_img, cv2.COLOR_BGR2GRAY)
        ax[i].imshow(rsz_img)
        ax[i].axis('off')

        aspct_img = crop_and_resize(img, target_w=224, target_h=224)
        aspct_img = cv2.cvtColor(aspct_img, cv2.COLOR_BGR2GRAY)
        ax[i + n_images].imshow(aspct_img)
        ax[i + n_images].axis('off')

        rgb_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
        ax[i + 2 * n_images].imshow(rgb_img)
        ax[i + 2 * n_images].axis('off')

        gray_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        ax[i + 3 * n_images].imshow(gray_img)
        ax[i + 3 * n_images].axis('off')

        gabor_img = cvt_gabor(gray_img)
        ax[i + 4 * n_images].imshow(gabor_img)
        ax[i + 4 * n_images].axis('off')

        sobel_img = sobel(gray_img)
        ax[i + 5 * n_images].imshow(sobel_img)
        ax[i + 5 * n_images].axis('off')

        canny_img = canny(gray_img, sigma=1.5)
        ax[i + 6 * n_images].imshow(canny_img)
        ax[i + 6 * n_images].axis('off')

    plt.subplots_adjust(hspace=0.05, wspace=0.05)

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

base_dir = '../FaceShape Dataset'
train_dir = os.path.join(base_dir, 'training_set')
test_dir = os.path.join(base_dir, 'testing_set')
categories = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
label_map = {0: 'Heart', 1: 'Oblong', 2: 'Oval', 3: 'Round', 4: 'Square'}
num_classes = len(categories)

if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Training directory not found: {train_dir}")

if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Test directory not found: {test_dir}")

training_data_apr = []
testing_data_apr = []

print("Processing Data with Aspect Ratio...")
create_data_files(train_dir, training_data_apr, 'aspect')
create_data_files(test_dir, testing_data_apr, 'aspect')

X_train_apr, X_test_apr, y_train_apr, y_test_apr = train_test_prep(training_data_apr, testing_data_apr)
print_summary(X_train_apr, X_test_apr, y_train_apr, y_test_apr)

counter = Counter([label for _, label in training_data_apr])
print(f"Class distribution in training dataset: {counter}")
counter = Counter([label for _, label in testing_data_apr])
print(f"Class distribution in test dataset: {counter}")

data_path = os.path.join(base_dir, 'pickles')
if not os.path.exists(data_path):
    os.makedirs(data_path)

joblib_out(X_train_apr, X_test_apr, y_train_apr, y_test_apr, version='apr', data_path=data_path)

print("Joblib files created successfully with Aspect Ratio.")

training_data_gray = []
testing_data_gray = []

print("Processing Data with Grayscale...")
create_data_files(train_dir, training_data_gray, 'gray')
create_data_files(test_dir, testing_data_gray, 'gray')

print(f'Training Images: {len(training_data_gray)}')
print(f'Testing Images: {len(testing_data_gray)}')

X_train_gray, X_test_gray, y_train_gray, y_test_gray = train_test_prep(training_data_gray, testing_data_gray)

print_summary(X_train_gray, X_test_gray, y_train_gray, y_test_gray)

counter = Counter([label for _, label in training_data_gray])
print(f"Class distribution in training dataset: {counter}")
counter = Counter([label for _, label in testing_data_gray])
print(f"Class distribution in test dataset: {counter}")

joblib_out(X_train_gray, X_test_gray, y_train_gray, y_test_gray, version='gray', data_path=data_path)

print("Joblib files created successfully with Grayscale.")

training_data_rgb = []
testing_data_rgb = []

print("Processing Data with RGB...")
create_data_files(train_dir, training_data_rgb, 'rgb')
create_data_files(test_dir, testing_data_rgb, 'rgb')

print(f'Training Images: {len(training_data_rgb)}')
print(f'Testing Images: {len(testing_data_rgb)}')

X_train_rgb, X_test_rgb, y_train_rgb, y_test_rgb = train_test_prep(training_data_rgb, testing_data_rgb)

print_summary(X_train_rgb, X_test_rgb, y_train_rgb, y_test_rgb)

counter = Counter([label for _, label in training_data_rgb])
print(f"Class distribution in training dataset: {counter}")
counter = Counter([label for _, label in testing_data_rgb])
print(f"Class distribution in test dataset: {counter}")

joblib_out(X_train_rgb, X_test_rgb, y_train_rgb, y_test_rgb, version='rgb', data_path=data_path)

print("Joblib files created successfully with RGB.")

print("Processing and displaying example images...")
display_preprocessed_images()

print("All data processed successfully and joblib files created.")
