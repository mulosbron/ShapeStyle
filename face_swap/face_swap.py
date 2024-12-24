import cv2
import dlib
import numpy as np
import os

# Yüz algılama ve işaretleyici modellerinin yüklenmesi
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Model dosyasının yolunu kontrol edin

# Görüntülerin yüklenmesi
user_image_path = r'C:\Users\duggy\OneDrive\Belgeler\Github\AIProject\face_swap\source.png'
target_image_path = r'C:\Users\duggy\OneDrive\Belgeler\Github\AIProject\face_swap\destination.png'

user_image = cv2.imread(user_image_path)
target_image = cv2.imread(target_image_path)

if user_image is None:
    print("Kullanıcı görseli yüklenemedi.")
    exit()
if target_image is None:
    print("Hedef görsel yüklenemedi.")
    exit()

# Gri tonlamaya dönüştürme
user_gray = cv2.cvtColor(user_image, cv2.COLOR_BGR2GRAY)
target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

# Yüz algılama
user_faces = detector(user_gray)
target_faces = detector(target_gray)

if len(user_faces) == 0:
    print("Kullanıcı görselinde yüz algılanamadı.")
    exit()
if len(target_faces) == 0:
    print("Hedef görselde yüz algılanamadı.")
    exit()


# Yüz işaret noktalarının çıkarılması
def get_landmarks(image, faces):
    return [predictor(image, face) for face in faces]


user_landmarks = get_landmarks(user_gray, user_faces)
target_landmarks = get_landmarks(target_gray, target_faces)


# İşaret noktalarını numpy array'e dönüştürme
def landmarks_to_np(landmarks):
    coords = np.zeros((68, 2), dtype=np.int32)
    for i in range(68):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    return coords


user_points = landmarks_to_np(user_landmarks[0])
target_points = landmarks_to_np(target_landmarks[0])


# Yüz hizalama için affine dönüşüm uygulama
def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return dst


# Delaunay üçgenleme için yüz sınırlarını belirleme
rect = cv2.boundingRect(np.float32(target_points))
subdiv = cv2.Subdiv2D(rect)

# Noktaları float olarak dönüştürüp ekleme
points = [tuple(map(float, p)) for p in target_points.tolist()]
for p in points:
    subdiv.insert(p)

# Üçgen listesi alınması
triangles = subdiv.getTriangleList()
triangles = np.array(triangles, dtype=np.int32)


# Fonksiyonlar
def rect_contains(rect, point):
    return rect[0] <= point[0] <= rect[0] + rect[2] and rect[1] <= point[1] <= rect[1] + rect[3]


def find_index(points, point):
    for i, p in enumerate(points):
        if abs(p[0] - point[0]) < 1 and abs(p[1] - point[1]) < 1:
            return i
    return -1


# Üçgenlerin indislerinin bulunması
triangle_indices = []
for t in triangles:
    pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
    indices = []
    for pt in pts:
        if rect_contains(rect, pt):
            idx = find_index(target_points, pt)
            if idx == -1:
                break
            indices.append(idx)
    if len(indices) == 3:
        triangle_indices.append(indices)


# Renk Düzeltme Fonksiyonu
def color_correction(source, target, mask):
    # Convert images to LAB color space
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Compute the mean and standard deviation of each channel
    source_mean, source_std = cv2.meanStdDev(source_lab, mask=mask)
    target_mean, target_std = cv2.meanStdDev(target_lab, mask=mask)

    # Subtract the mean from the source
    source_lab -= source_mean.reshape((1, 1, 3))

    # Scale by the standard deviation ratio
    source_lab = (source_lab * (target_std.reshape((1, 1, 3)) / source_std.reshape((1, 1, 3))))

    # Add the target mean
    source_lab += target_mean.reshape((1, 1, 3))

    # Clip the values to [0, 255] and convert back to uint8
    source_lab = np.clip(source_lab, 0, 255).astype(np.uint8)

    # Convert back to BGR
    corrected = cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)

    return corrected


# Hedef maskesi oluşturma
mask = np.zeros(target_gray.shape, dtype=np.uint8)
cv2.fillConvexPoly(mask, cv2.convexHull(target_points), 255)

# Üçgenlerin işlenmesi
for tri in triangle_indices:
    src_tri = [user_points[tri[i]] for i in range(3)]
    dst_tri = [target_points[tri[i]] for i in range(3)]

    # Affine dönüşüm uygulanması
    src_tri_np = np.float32(src_tri)
    dst_tri_np = np.float32(dst_tri)

    # Bounding rect hesaplanması
    src_rect = cv2.boundingRect(src_tri_np)
    dst_rect = cv2.boundingRect(dst_tri_np)

    # Bölge noktalarının kaydırılması
    src_tri_rect = []
    dst_tri_rect = []
    for i in range(3):
        src_tri_rect.append(((src_tri_np[i][0] - src_rect[0]), (src_tri_np[i][1] - src_rect[1])))
        dst_tri_rect.append(((dst_tri_np[i][0] - dst_rect[0]), (dst_tri_np[i][1] - dst_rect[1])))

    # Kaynak bölgenin kırpılması
    src_cropped = user_image[src_rect[1]:src_rect[1] + src_rect[3], src_rect[0]:src_rect[0] + src_rect[2]]

    # Affine dönüşümün uygulanması
    warped_tri = apply_affine_transform(src_cropped, src_tri_rect, dst_tri_rect, (dst_rect[2], dst_rect[3]))

    # Üçgen maskesinin oluşturulması
    mask_tri = np.zeros((dst_rect[3], dst_rect[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask_tri, np.int32(dst_tri_rect), (1.0, 1.0, 1.0), 16, 0)

    # Renk Düzeltme Uygulama
    mask_tri_gray = cv2.cvtColor((mask_tri * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    corrected_warped_tri = color_correction(warped_tri, target_image[dst_rect[1]:dst_rect[1] + dst_rect[3],
                                                        dst_rect[0]:dst_rect[0] + dst_rect[2]], mask_tri_gray)

    # Hedef görüntüye ekleme
    target_image_region = target_image[dst_rect[1]:dst_rect[1] + dst_rect[3], dst_rect[0]:dst_rect[0] + dst_rect[2]]
    target_image[dst_rect[1]:dst_rect[1] + dst_rect[3], dst_rect[0]:dst_rect[0] + dst_rect[2]] = \
        target_image_region * (1 - mask_tri) + corrected_warped_tri * mask_tri

# Son birleşim için mask oluşturma
final_mask = np.zeros(target_gray.shape, dtype=np.uint8)
cv2.fillConvexPoly(final_mask, cv2.convexHull(target_points), 255)

# Seamless cloning için cilt tonlarını uyumlu hale getirmek adına renk düzeltme uygulama
result = cv2.seamlessClone(target_image, target_image, final_mask, (rect[0] + rect[2] // 2, rect[1] + rect[3] // 2),
                           cv2.NORMAL_CLONE)

# Sonucun kaydedilmesi
output_path = r'C:\Users\duggy\OneDrive\Belgeler\Github\AIProject\face_swap\swapped_fr_corrected.jpg'
cv2.imwrite(output_path, result)
print(f"Yüz değiştirme tamamlandı. Sonuç kaydedildi: {output_path}")

# Sonucun gösterilmesi (isteğe bağlı)
# cv2.imshow('Sonuç', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
