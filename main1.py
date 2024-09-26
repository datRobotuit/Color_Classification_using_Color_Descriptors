import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
import os
import glob

# Hàm tính Histogram
def compute_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Hàm tính Color Moments (mean, variance, skewness)
def compute_color_moments(image):
    moments = []
    for i in range(3):  # For each channel (B, G, R)
        channel = image[:, :, i]
        moments.append(np.mean(channel))
        moments.append(np.std(channel))
        moments.append(scipy.stats.skew(channel.flatten()))
    return moments

# Hàm tính CDC và CCV cần được cài đặt thêm theo tài liệu
# Phần này sẽ tuỳ thuộc vào cách mô tả chi tiết của hai loại đặc trưng này

# Load hình ảnh và tính đặc trưng
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in glob.glob(os.path.join(folder, '*.jpg')):  # Load ảnh từ thư mục
        img = cv2.imread(filename)
        if img is not None:
            # Tính đặc trưng của ảnh
            hist = compute_histogram(img)
            color_moments = compute_color_moments(img)
            # CDC và CCV cần cài đặt thêm
            features = np.concatenate([hist, color_moments])
            images.append(features)
            labels.append(int(os.path.basename(filename).split('_')[0]))  # Giả sử nhãn nằm trong tên file
    return np.array(images), np.array(labels)

# Load dữ liệu train và test
train_data, train_labels = load_images_from_folder('train')
test_data, test_labels = load_images_from_folder('test')

# Khởi tạo KNN với k = 1 và k = 5
for k in [1, 5]:
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')  # Có thể thay metric bằng các độ đo khác
    knn.fit(train_data, train_labels)
    
    # Dự đoán
    test_pred = knn.predict(test_data)
    
    # Đánh giá độ chính xác
    accuracy = accuracy_score(test_labels, test_pred)
    print(f'Accuracy with k={k}: {accuracy}')