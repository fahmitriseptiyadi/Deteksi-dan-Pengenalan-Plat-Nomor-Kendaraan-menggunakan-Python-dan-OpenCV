import cv2
import pytesseract      #library ini untuk mengekstrak teks pelat nomor dari pelat nomor yang terdeteksi.
import imutils  #library ini untuk mengubah ukuran gambar kita.
import csv

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract';

# Mengambil input gambar kami
# mengubah ukuran lebarnya menjadi 500 piksel width
image = cv2.imread("test3.jpg")
image = imutils.resize(image, width=500)
cv2.imshow("Original Image", image)
cv2.waitKey(0)

# Mengubah gambar input menjadi grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("greyed image", gray_image)
cv2.waitKey(0)

# Mengurangi noise pada Grayscale
gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17)
cv2.imshow("smoothened image", gray_image)
cv2.waitKey(0)

# Mendeteksi tepi gambar yang dihaluskan
edged = cv2.Canny(gray_image, 30, 200)
cv2.imshow("edged image", edged)
cv2.waitKey(0)

# Menemukan kontur dari gambar bermata
# RETR_LIST: Ini mengambil semua kontur tetapi tidak terhubung
# CHAIN_APPROX_SIMPLE: Menghapus semua titik redundan pada kontur yang terdeteksi.
# Kami membuat salinan dari gambar masukan asli. tidak mengubah gambar asli.
cnts,new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
image1=image.copy()
cv2.drawContours(image1,cnts,-1,(0,255,0),3)
cv2.imshow("contours",image1)
cv2.waitKey(0)

# mengurutkan kontur berdasarkan area minimum 30 dan mengabaikan yang di bawahnya.
# screenCnt = None: Menyimpan kontur plat nomor.
# Menampilkan gambar yang berisi 30 kontur teratas yang digambar di sekitarnya.
cnts = sorted(cnts, key = cv2.contourArea, reverse = True) [:30]
# screenCnt = None
image2 = image.copy()
cv2.drawContours(image2,cnts,-1,(0,255,0),3)
cv2.imshow("Top 30 contours",image2)
cv2.waitKey(0)

# membuat loop for di atas kontur yang telah disortir. Ini untuk menemukan kontur terbaik dari pelat nomor yang kami harapkan.

# # x,y,w,h = cv2.boundingRect(c) : Ini menemukan koordinat bagian yang diidentifikasi sebagai plat nomor.
# # cv2.imwrite('./'+str(i)+'.png',new_img: Menyimpan gambar baru dari pelat nomor yang dipotong.
i=7
for c in cnts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
        if len(approx) == 4:
                screenCnt = approx
        # menemukan koordinat bagian yang diidentifikasi sebagai plat nomor.
        x,y,w,h = cv2.boundingRect(c)
        new_img=image[y:y+h,x:x+w]
        # Menyimpan gambar baru dari pelat nomor yang dipotong.
        cv2.imwrite('./'+str(i)+'.png',new_img)
        i+=1
        break

# menggambar kontur yang dipilih untuk menjadi pelat nomor pada gambar asli
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
cv2.imshow("image with detected license plate", image)
cv2.waitKey(0)

# menampilkan gambar bagian plat nomor yang dipotong.
# memanggil pytesseract untuk mengekstrak teks pada gambar.
Cropped_loc = './7.png'
cv2.imshow("cropped", cv2.imread(Cropped_loc))
plate = pytesseract.image_to_string(Cropped_loc, lang='eng')
print("Number plate is:", plate)
cv2.waitKey(0)
cv2.destroyAllWindows()