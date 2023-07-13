# Import các thư viện cần thiết:
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Tạo các biến để lưu đường dẫn tới các thư mục chứa dữ liệu:
train_dir = 'C:/Users/DELL/Downloads/Nhom09_De06_BenhVeLaCuaTao/Benh_La_Tao/Slipting_data/training'
val_dir = 'C:/Users/DELL/Downloads/Nhom09_De06_BenhVeLaCuaTao/Benh_La_Tao/Slipting_data/validation'

# Khởi tạo các thông số cho model và quá trình train:
batch_size = 32
img_height = 224
img_width = 224
num_epochs = 200
learning_rate = 0.001 # tốc độ học

# Sử dụng ImageDataGenerator để đọc và tiền xử lý dữ liệu ảnh:
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20, # Xoay ảnh trong khoảng [-20, 20] độ
                                   width_shift_range=0.1, # Cắt ảnh theo chiều rộng trong khoảng [-0.1, 0.1]
                                   height_shift_range=0.1, # Cắt ảnh theo chiều cao trong khoảng [-0.1, 0.1]
                                   shear_range=0.1, # Cho phép cắt hình ảnh với một góc cắt tối đa là 10%
                                   zoom_range=0.1,  # Phóng to hoặc thu nhỏ ảnh trong khoảng [0.9, 1.1]
                                   vertical_flip=False, # Không lật ảnh theo chiều dọc
                                   horizontal_flip=True, # Lật ảnh theo chiều ngang
                                   fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)


# Sử dụng flow_from_directory để tạo ra các bộ dữ liệu train và validation:
train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size=(img_height, img_width),# Resize ảnh về kích thước224x224
                                               batch_size=batch_size,
                                               class_mode='categorical')

val_data = val_datagen.flow_from_directory(val_dir,
                                           target_size=(img_height, img_width),# Resize ảnh về kích thước224x224
                                           batch_size=batch_size,
                                           class_mode='categorical')

# # Khởi tạo object ImageDataGenerator
# datagen = ImageDataGenerator(
#     rotation_range=20,
#     horizontal_flip=True, 
#     vertical_flip=False, # Không lật ảnh theo chiều dọc
#     width_shift_range=0.1, 
#     height_shift_range=0.1, 
#     zoom_range=0.1, # Phóng to hoặc thu nhỏ ảnh trong khoảng [0.9, 1.1]
# )
# 
# # Áp dụng data augmentation trên tập huấn luyện
# train_datagen = datagen.flow_from_directory(
#     train_dir,
#     target_size=(img_height, img_width), # Resize ảnh về kích thước224x224
#     batch_size=batch_size,
#     class_mode='categorical'
# )

# Xây dựng model CNN:
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile model và bắt đầu quá trình train:
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train_data,
          epochs=num_epochs,
          validation_data=val_data)
        #   ,initial_epoch=15790)



# Tính độ chính xác trên tập validation và in ra kết quả
accuracy = model.evaluate(val_data)[1]
print("Validation accuracy: {:.2f}%".format(accuracy * 100))

# Lưu model
model.save('model200.h5')
