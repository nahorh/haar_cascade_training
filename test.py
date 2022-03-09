import cv2

cascade = cv2.CascadeClassifier('./cascade.xml')

perfect_img = cv2.imread('./testimages/perfect.png', 1)
perfect_img = cv2.resize(perfect_img, (600, 400))
test_img = cv2.imread('./testimages/perfect.png', 0)

obj1 = cascade.detectMultiScale(perfect_img, scaleFactor=1.001,
                                minNeighbors=6, minSize=(180, 120), maxSize=(180, 120))


for (x, y, w, h) in obj1:
    cv2.rectangle(perfect_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
cv2.imshow('img', perfect_img)
cv2.waitKey(0)


def usecam():
    vid = cv2.VideoCapture(0)
    while True:
        ret, img = vid.read()
        img = cv2.resize(img, (600, 400))
        obj = cascade.detectMultiScale(img, scaleFactor=1.001,
                                       minNeighbors=6, minSize=(180, 120), maxSize=(180, 120))
        for (x, y, w, h) in obj:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow('img', img)
        if cv2.waitKey(1) & ord('q') == 'q':
            cv2.destroyAllWindows()
            break
