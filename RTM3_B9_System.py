import cv2
import pytesseract      #library ini untuk mengekstrak teks pelat nomor dari pelat nomor yang terdeteksi.
import imutils  #library ini untuk mengubah ukuran gambar kita.
import csv
import math
import sys
import numpy as np
import pytesseract
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PIL import ImageQt
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract';

#fungsi konvolusi
def conv(X, F):
    X_height = X.shape[0]
    X_width = X.shape[1]

    F_height = F.shape[0]
    F_width = F.shape[1]

    H = (F_height) // 2
    W = (F_width) // 2

    out = np.zeros((X_height, X_width))
    for i in np.arange(H + 1, X_height - H):
        for j in np.arange(W + 1, X_width - W):
            sum = 0
            for k in np.arange(-H, H + 1):
                for l in np.arange(-W, W + 1):
                    a = X[i + k, j + l]
                    w = F[H + k, W + l]
                    sum += (w * a)
            out[i, j] = sum
    return out

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('RTM3.ui', self)
        self.Image = None
        # screenCnt = None


        # self.button_loadcitra.clicked.connect(self.fungsi)

        # OPEN
        self.btnLoadCitra.clicked.connect(self.open)

        self.trackbarBrightness.valueChanged[int].connect(self.brightness)
        self.trackbarContrast.valueChanged[int].connect(self.contrast)

        # CROP
        self.btnCrop.clicked.connect(self.crop)

        # CROP
        self.btnReset.clicked.connect(self.reset)

        # SAVE
        self.btnSave.clicked.connect(self.save)

        # GRAYSCALE
        self.btnGrayscale.clicked.connect(self.grayscale)

        # SMOOTH
        self.btnSmooth.clicked.connect(self.smooth)

        # EDGE
        self.btnDetect.clicked.connect(self.detect)

        # GEOMETRY OPERATION
        self.actionTranslation.triggered.connect(self.translasi)
        self.actionTranspose.triggered.connect(self.transpose)

        # SPATIAL
        self.action2D_Convolution.triggered.connect(self.convsecondD)
        self.actionConvKernel_I.triggered.connect(self.convfilteringI)
        self.actionConvKernel_II.triggered.connect(self.convfilteringII)
        self.actionMedianFiltering.triggered.connect(self.medianfiltering)
        self.actionGaussian.triggered.connect(self.gaussian)
        self.actionSharpening_Laplace.triggered.connect(self.sharpeninglaplace)

        # HISTOGRAM
        self.actionHGrayscale.triggered.connect(self.histgrayscale)
        self.actionHRGB.triggered.connect(self.histrgb)
        self.actionHEqualization.triggered.connect(self.histequalization)
    def open(self):
        filename, filter = QFileDialog.getOpenFileName(self, 'Open File','D:\Itenas\MataKuliah\SMT 4\PENGOLAHAN CITRA DIGITAL PRAKTIKUM\ProgramPCD\TestImagePlat',"Image Files(*.jpg)""\nImage Files(*.jpeg)""\nImage Files(*.png)")
        if filename:
            self.Image = cv2.imread(filename)
            self.displayImage(1)
            self.labelOriginalImage.setText("Original Image")
        else:
            print('Gagal Memuat')
        self.labelDirektori.setText(filename)

    def crop(self):
        image = self.Image #memanggil citra dan dimasukkan ke dalam variabel image
        roi = cv2.selectROI(image) #menggunakan fungsi ROI pada cv2 untuk menentukan daerah yang ingin dilakukan proses crop
        print (roi) #menampilkan jendela ROI
        image_cropped = image[int(roi[1]):int(roi[1] + roi[3]), #proses pemotongan citra sesuai inputan ROI
                        int(roi[0]):int(roi[0] + roi[2])]
        resize_image = cv2.resize(image_cropped, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC) #proses resize citra setelah dilakukan pemotongan
        self.Image = resize_image #hasil resize citra dimasukkan ke dalam variabel
        self.displayImage(1) #menampilkan citra

    def save(self):
        filename, filter = QFileDialog.getSaveFileName(self, 'Save File','D:\Itenas\MataKuliah\SMT 4\PENGOLAHAN CITRA DIGITAL PRAKTIKUM\ProgramPCD\TestImagePlat\Result',"JPG Image (*.jpg)")
        if filename:
            cv2.imwrite(filename, self.Image)
        else:
            print('Tidak Dapat Menyimpan')

    def reset(self):
        self.labelInput.clear()
        self.labelHasil.clear()
        self.labelDetect.clear()
        self.labelDetect2.clear()
        self.labelCroppedImage.clear()
        self.labelEdgeImage.clear()
        self.labelTopContourImage.clear()
        self.labelOriginalImage.setText("")
        self.labelSmoothenedImage.setText("")
        self.labelGrayscaleImage.setText("")
        self.labelTopContourImage.setText("")
        self.labelDirektori.setText("")
        self.labelDeteksiPlatHasil.setText("")

    def brightness(self, value):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass
        # print('\n', 'Nilai Pixel Citra : ')
        # print(self.Image)

        H, W = self.Image.shape[:2]# mengambil Height&Weight sebuah citra yang sudah di grayscale
        brightness = value + 30 # penambhan kecerahan menggunakan trackbar
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = np.clip(a + brightness, 0, 255)# proses perhitungan brightness dengan citra asli

                self.Image.itemset((i, j), b)

        self.displayImage(2)
        # print('\n', 'Nilai Pixel Brightness : ')
        # print(self.Image)

    def contrast(self, value):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass
        print('\n', 'Nilai Pixel Citra : ')
        print(self.Image)

        H, W = self.Image.shape[:2]#mengambil Height&Weight sebuah citra yang sudah di grayscale
        contrast = value + 0.7 # jika ingin melakukan penambhan contrast menggunakan trackbar
        for i in range(H): # perulanangan height
            for j in range(W):
                a = self.Image.item(i, j)
                b = np.clip(a * contrast, 0, 255) # proses perhitungan kontras dengan citra asli

                self.Image.itemset((i, j), b)

        self.displayImage(2)
        # print('\n', 'Nilai Pixel Contrast : ')
        # print(self.Image)

    def translasi(self):
        h, w = self.Image.shape[:2] #memanggil bentuk citra
        quarter_h, quarter_w = h / 8, w / 8
        T = np.float32([[1, 0, quarter_w], [0, 1, quarter_h]])
        #proses translasi
        img = cv2.warpAffine(self.Image, T, (w, h)) #memasukan proses translasi
        self.Image = img #memanggil proses translasi
        self.displayImage(1) #menampilkan pada window 1

    def transpose(self):
        img = self.Image
        transpose_img = cv2.transpose(img)
        self.Image = transpose_img
        self.displayImage(1)  # menampilkan pada window 1


    def convsecondD(self):
        img = self.Image
        kernel = np.ones((5, 5), np.float32) / 25
        smooth = cv2.filter2D(img, -1, kernel)
        self.Image = smooth
        self.displayImage(2)
        self.labelGrayscaleImage.setText("")
        self.labelSmoothenedImage.setText("Smoothened Image 2D Convolution")
        print('\n', 'Nilai Pixel Smooth : ')
        print(self.Image)

    def sharpeninglaplace(self):
        img = self.Image
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
        self.Image = image_sharp
        self.displayImage(2)
        self.labelGrayscaleImage.setText("")
        self.labelSmoothenedImage.setText("Smoothened Image Sharpening")
        print('\n', 'Nilai Pixel Smooth : ')
        print(self.Image)


    def convfilteringI(self):
        img = self.Image
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1], ], dtype='float')
        img_out = conv(image, kernel)
        # plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def convfilteringII(self):
        img = self.Image
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.array(
            [[6, 0, -6],
             [6, 1, -6],
             [6, 0, -6], ], dtype='float')
        img_out = conv(image, kernel)
        plt.imshow(img_out, cmap='original', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def medianfiltering(self):
        image = self.Image
        # grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        smooth = cv2.medianBlur(image, 5)
        self.Image = smooth
        self.displayImage(2)
        self.labelGrayscaleImage.setText("")
        self.labelSmoothenedImage.setText("Smoothened Image ""Median""")
        print('\n', 'Nilai Pixel Smooth : ')
        print(self.Image)


    def gaussian(self):
        image = self.Image
        # grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        smooth = cv2.GaussianBlur(image, (5,5),0)
        self.Image = smooth
        self.displayImage(2)
        self.labelGrayscaleImage.setText("")
        self.labelSmoothenedImage.setText("Smoothened Image ""Gaussian""")
        print('\n', 'Nilai Pixel Smooth : ')
        print(self.Image)

    @pyqtSlot()
    def histgrayscale(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255)

        self.Image = gray
        print(self.Image)
        self.displayImage(2)
        plt.hist(self.Image.ravel(), 255, [0, 255])
        plt.show()

    @pyqtSlot()
    def histrgb(self):
        color = ('b', 'g', 'r')  # inisialisasi variabel warna
        for i, col in enumerate(color):  # memulai looping untuk membaca nilai rgb
            histo = cv2.calcHist([self.Image], [i], None, [256], [0, 256])
            # Membuat variabel untuk menghitung nilai r,g, dan b
            plt.plot(histo, color=col)  # menampilkan variabel histo ditambah dengan warna
            plt.xlim([0, 256])  # Merupakan format warnanya
        self.displayImage(2)
        plt.show()  # untuk menampilkan grafik


    @pyqtSlot()
    def histequalization(self):  # nama prosedur
        hist, bins = np.histogram(self.Image.flatten(), 256, [0, 256])
        # inisialisasi hist dan bins untuk menampilkan histogram dengan skala maks 256
        cdf = hist.cumsum()  # inisialisasi cdf untuk membuat grafik cdf
        cdf_normalized = cdf * hist.max() / cdf.max()  # rumus proses cdf normalisasi
        cdf_m = np.ma.masked_equal(cdf, 0)  # proses masking
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())  # rumus perataan citra
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')  # menginputkan nilai cdf
        self.Image = cdf[self.Image]  # memasukan fungsi cdf ke self.image
        self.displayImage(2)  # menampilkan pada jendela kedua

        plt.plot(cdf_normalized, color='b')  # untuk normaliasisi warna biru
        plt.hist(self.Image.flatten(), 256, [0, 256], color='r')  # membuat histogram warna merah maks 256
        plt.xlim([0, 256])  # membuat nilai limit x 0 sampai 256
        plt.legend(('cdf', 'histogram'), loc='upper left')  # membuat window menunjukan keterangan
        plt.show()  # untuk menampilkan histogram


    def grayscale(self):
        image = self.Image
        grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.Image = grayimage
        self.displayImage(2)
        self.labelSmoothenedImage.setText("")
        self.labelGrayscaleImage.setText("Grayscale Image")
        print('\n', 'Nilai Pixel Grayscale : ')
        print(self.Image)

        # self.btnSmooth.setEnabled(True)
        # self.groupBright.setEnabled(True)
        # self.groupCont.setEnabled(True)
        # self.Button_edgeDet.setEnabled(True)
        # self.Button_reset.setEnabled(True)
        # self.button_simpan.setEnabled(True)
        # self.Button_gaussian.setEnabled(True)
    def smooth(self):
        # self.btnGrayscale.setEnabled(True)
        image = self.Image
        # grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        smooth = cv2.bilateralFilter(image, 11, 17, 17)
        self.Image = smooth
        self.displayImage(2)
        self.labelGrayscaleImage.setText("")
        self.labelSmoothenedImage.setText("Smoothened Image ""Billateral""")
        print('\n', 'Nilai Pixel Smooth : ')
        print(self.Image)

    # edge detection menggunakan canny
    # karena ketika menggunakan sobel atau prewwit hasil nya kurang memuaskan sehingga mempengaruhi pencarian contour
    def detect(self):
        image = self.Image
        # Mendeteksi tepi gambar yang dihaluskan
        edged = cv2.Canny(image, 30, 200)
        self.Image = edged
        self.displayImage(3)
        self.labelEdgeImage.setText("Edge Detection Image")

        # Menemukan kontur dari gambar bermata
        # RETR_LIST: Ini mengambil semua kontur tetapi tidak terhubung
        # CHAIN_APPROX_SIMPLE: Menghapus semua titik redundan pada kontur yang terdeteksi.
        # Kami membuat salinan dari gambar masukan asli. tidak mengubah gambar asli.

        image2 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour_image = image2.copy()
        cv2.drawContours(contour_image, cnts, -1, (0, 255, 0), 3)
        # cv2.imshow("contours", contour_image)
        # cv2.waitKey(0)

        # mengurutkan kontur berdasarkan area minimum 30 dan mengabaikan yang di bawahnya.
        # screenCnt = None: Menyimpan kontur plat nomor.
        # Menampilkan gambar yang berisi 30 kontur teratas yang digambar di sekitarnya.

        image3 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
        screenCnt = None
        top30contours_image = image3.copy()
        cv2.drawContours(top30contours_image, cnts, -1, (0, 255, 0), 3)
        # cv2.imshow("Top 30 contours", top30contours_image)
        # cv2.waitKey(0)

        # membuat loop for di atas kontur yang telah disortir. Ini untuk menemukan kontur terbaik dari pelat nomor yang kami harapkan.

        # x,y,w,h = cv2.boundingRect(c) : Ini menemukan koordinat bagian yang diidentifikasi sebagai plat nomor.
        # cv2.imwrite('./'+str(i)+'.png',new_img: Menyimpan gambar baru dari pelat nomor yang dipotong.
        i = 7
        for c in cnts:
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
            if len(approx) == 4:
                screenCnt = approx
            # menemukan koordinat bagian yang diidentifikasi sebagai plat nomor.
            x, y, w, h = cv2.boundingRect(c)
            new_img = image[y:y + h, x:x + w]
            # Menyimpan gambar baru dari pelat nomor yang dipotong.
            cv2.imwrite('./' + str(i) + '.png', new_img)
            i += 1
            break

        # menggambar kontur yang dipilih untuk menjadi pelat nomor pada gambar asli
        image4 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(image4, [screenCnt], -1, (0, 255, 0), 3)
        self.Image = image4
        self.displayImage(4)
        self.labelTopContourImage.setText("Plate Contour Image")
        # menampilkan gambar bagian plat nomor yang dipotong.
        # memanggil pytesseract untuk mengekstrak teks pada gambar.
        Cropped_loc = './7.png'
        imageCropped = cv2.imread(Cropped_loc)
        self.Image = imageCropped
        self.displayImage(5)
        # cv2.imshow("cropped", cv2.imread(Cropped_loc))
        plate = pytesseract.image_to_string(Cropped_loc, lang='eng')

        print("Number plate is:", plate)
        self.labelDeteksiPlatHasil.setText(plate)


    def displayImage(self, windows=1):
        qformat = QImage.Format_Indexed8

        if len(self.Image.shape) == 3:
            if (self.Image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.Image, self.Image.shape[1], self.Image.shape[0],
                     self.Image.strides[0], qformat)
        img = img.rgbSwapped()

        if windows == 1:
            self.labelInput.setPixmap(QPixmap.fromImage(img))
            self.labelInput.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.labelInput.setScaledContents(True)
        if windows == 2:
            self.labelHasil.setPixmap(QPixmap.fromImage(img))
            self.labelHasil.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.labelHasil.setScaledContents(True)

        if windows == 3:
            self.labelDetect.setPixmap(QPixmap.fromImage(img))
            self.labelDetect.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.labelDetect.setScaledContents(True)

        if windows == 4:
            self.labelDetect2.setPixmap(QPixmap.fromImage(img))
            self.labelDetect2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.labelDetect2.setScaledContents(True)

        if windows == 5:
            self.labelCroppedImage.setPixmap(QPixmap.fromImage(img))
            self.labelCroppedImage.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.labelCroppedImage.setScaledContents(True)


app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('RTM3')
window.show()
sys.exit(app.exec_())