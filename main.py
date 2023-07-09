
import random 
import shutil
import cv2
import numpy as np 
import pandas as pd 
import os
from PyQt5 import QtCore, QtGui, QtWidgets
import tensorflow as tf
from keras.models import load_model
SEED=1

class Ui_MainWindow(object):  
    
    
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(472, 672)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, 10, 391, 61))
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(50, 560, 121, 51))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(310, 560, 121, 51))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(60, 90, 361, 341))
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(80, 450, 321, 41))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(80, 500, 321, 41))
        self.label_4.setObjectName("label_4")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 472, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        
        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        #self.label.setText(_translate("MainWindow", "<html><head/><body><p align=/"center/"><span style=/" font-size:16pt; font-weight:600;/">DỰ ĐOÁN BỆNH Ở LÁ TÁO</span></p></body></html>"))
        self.pushButton.setText(_translate("MainWindow", "Chọn File Ảnh"))
        self.pushButton_2.setText(_translate("MainWindow", "Dự Đoán"))
        self.label_3.setText(_translate("MainWindow", ""))
        self.label_4.setText(_translate("MainWindow", "Dự đoán ảnh....."))
        


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.choose_file)
        self.pushButton_2.clicked.connect(self.predict_image)
        
        
        # Đặt đường dẫn đến thư mục chứa dữ liệu
        data_dir = 'C:/Users/DELL/Downloads/Nhom09_De06_BenhVeLaCuaTao/Benh_La_Tao'

        # Di chuyển đến thư mục chứa dữ liệu
        os.chdir(data_dir)

        # Kiểm tra số lượng ảnh trong từng thư mục
        print(len(os.listdir('Apple_cedar_rust')))
        print(len(os.listdir('Apple_scab')))
        print(len(os.listdir('Apple_black_rot')))

        # Tạo thư mục mới để chứa tập huấn luyện và tập validation
        base_dir='C:/Users/DELL/Downloads/Nhom09_De06_BenhVeLaCuaTao/Benh_La_Tao/Slipting_data'
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        os.chdir(base_dir)

        # Tạo các thư mục con trong tập huấn luyện và tập validation
        subdirs = ['training', 'validation']
        classes = ['Apple_cedar_rust', 'Apple_scab', 'Apple_black_rot']
        for subdir in subdirs:
            subdir_path = os.path.join(base_dir, subdir)
            if not os.path.exists(subdir_path):
                os.mkdir(subdir_path)
            for cls in classes:
                cls_path = os.path.join(subdir_path, cls)
                if not os.path.exists(cls_path):
                    os.mkdir(cls_path)

        # Kiểm tra các thư mục đã được tạo thành công
        print(os.listdir(base_dir + '/training'))
        print(os.listdir(base_dir + '/validation'))



        # Chia tập dữ liệu thành tập huấn luyện và tập validation
        for item in classes:
            n_val = round(len(os.listdir(os.path.join(data_dir, item))) * 0.2)
            n_train = round(len(os.listdir(os.path.join(data_dir, item))) * 0.8)
            fnames = os.listdir(os.path.join(data_dir, item))
            assert(n_val + n_train == len(fnames))

            # Xáo trộn dữ liệu và chia thành tập huấn luyện và tập validation
            random.seed(SEED+5)
            random.shuffle(fnames)
            val_fnames = fnames[0:n_val]
            tr_fnames = fnames[n_val:len(fnames)]
            assert(len(val_fnames) + len(tr_fnames) == len(fnames))

            for i in val_fnames:
                # Đường dẫn đến file ảnh trong thư mục gốc
                src = os.path.join(data_dir, item, i)
                # Đường dẫn đến file ảnh trong thư mục validation
                dest = os.path.join(base_dir, 'validation', item, i)
                shutil.copy(src, dest)

            for j in tr_fnames:
                # Đường dẫn đến file ảnh trong thư mục gốc
                src = os.path.join(data_dir, item, j)
                # Đường dẫn đến file ảnh trong thư mục training
                dest = os.path.join(base_dir, 'training', item, j)
                shutil.copy(src, dest)

        # In ra số lượng ảnh trong từng thư mục của tập huấn luyện và tập validation
        for i in classes:
            path = os.path.join(base_dir, 'training', i)
            print('Training samples in {} is {}'.format(i, len(os.listdir(path))))

            path = os.path.join(base_dir, 'validation', i)
            print('Validation samples in {} is {}/n'.format(i, len(os.listdir(path))))

    def choose_file(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self,"Select Image File", "","Image Files (*.png *.jpg *.jpeg *.bmp)", options=options)
        if fileName:
            pixmap = QtGui.QPixmap(fileName)
            pixmap = pixmap.scaled(361, 341, QtCore.Qt.KeepAspectRatio)
            self.label_2.setPixmap(pixmap)
            self.label_3.setText("Image loaded successfully!")
        else:
            self.label_3.setText("No image selected!")

    def predict_image(self):
        
        model_path = 'C:/Users/DELL/Downloads/Nhom09_De06_BenhVeLaCuaTao/Benh_La_Tao/model200.h5'
        model = load_model(model_path)
        self.label_4.setText("Prediction in progress...")
        
        # Đọc ảnh từ label
        img = self.label_2.pixmap().toImage()
        # Chuyển đổi định dạng ảnh từ QImage sang numpy array
        buffer = QtCore.QBuffer()
        buffer.open(QtCore.QIODevice.ReadWrite)
        img.save(buffer, "PNG")
        data = np.frombuffer(buffer.data(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)

        # Chuyển đổi định dạng màu từ RGB sang BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Thay đổi kích thước ảnh
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

        # Đưa ảnh vào mạng nơ-ron (model đã huấn luyện sẵn) để dự đoán
        predictions = model.predict(np.array([img]))

        # Lấy kết quả dự đoán và hiển thị lên giao diện
        predicted_class = np.argmax(predictions, axis=-1)[0]
        if predicted_class == 0:
            self.label_4.setText("Dự đoán: Apple black rot")
        elif predicted_class == 1:
            self.label_4.setText("Dự đoán: Apple cedar rust")
        else:
            self.label_4.setText("Dự đoán: Apple scab")
            
            
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = MainWindow()
    MainWindow.show()
    sys.exit(app.exec_())
    