import os
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras_vggface.vggface import VGGFace
from PIL import ImageFile
import warnings
from joblib import load

warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True

data_path = r"C:\Users\duggy\OneDrive\Belgeler\Github\AIProject\FaceShape Dataset\pickles"
X_train_rgb = load(os.path.join(data_path, 'X_train_rgb.joblib'))
y_train_rgb = load(os.path.join(data_path, 'y_train_rgb.joblib'))
X_test_rgb = load(os.path.join(data_path, 'X_test_rgb.joblib'))
y_test_rgb = load(os.path.join(data_path, 'y_test_rgb.joblib'))

print(f"X_train_rgb shape: {X_train_rgb.shape}")
print(f"y_train_rgb shape: {y_train_rgb.shape}")
print(f"X_test_rgb shape: {X_test_rgb.shape}")
print(f"y_test_rgb shape: {y_test_rgb.shape}")

num_classes = 5

model_path = "face_shape_model_vgg16_rgb.h5"
learning_rate = 0.0001
epochs = 300

base_model = VGGFace(
    model="vgg16",
    include_top=False,
    input_shape=(224, 224, 3),
    pooling="avg"
)

for layer in base_model.layers:
    layer.trainable = False

if os.path.exists(model_path):
    print("Mevcut model bulunuyor. Model yükleniyor ve eğitime devam ediliyor...")
    model = load_model(model_path)
    for layer in base_model.layers:
        layer.trainable = False
    for layer in model.layers[:-2]:
        layer.trainable = False
else:
    print("Model bulunamadı. Yeni model oluşturuluyor...")
    x = base_model.output
    x = Dense(1028, activation="relu", name="fc1")(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation="softmax", name="classifier")(x)
    model = Model(inputs=base_model.input, outputs=output)
    for layer in base_model.layers:
        layer.trainable = False

model.compile(
    optimizer=Adam(lr=learning_rate),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

history = model.fit(
    X_train_rgb,
    y_train_rgb,
    epochs=epochs,
    batch_size=32,
    validation_data=(X_test_rgb, y_test_rgb)
)

model.save(model_path)
print(f"Model kaydedildi: {model_path}")


def plot_results(history, metric):
    train_metric = history.history.get(metric)
    val_metric = history.history.get(f"val_{metric}")
    if train_metric is not None:
        plt.plot(train_metric, label=f"Training {metric}")
    else:
        print(f"Training metric '{metric}' not found in history.")
    if val_metric is not None:
        plt.plot(val_metric, label=f"Validation {metric}")
    else:
        print(f"Validation metric 'val_{metric}' not found in history.")
    plt.xlabel("Epochs")
    plt.ylabel(metric.capitalize())
    plt.title(f"Training and Validation {metric.capitalize()}")
    plt.legend()
    plt.show()


plot_results(history, "loss")
plot_results(history, "acc")

loss, accuracy = model.evaluate(X_test_rgb, y_test_rgb)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
