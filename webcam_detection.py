# источник https://pyimagesearch.com/2017/09/18/real-time-object-detection-with-deep-learning-and-opencv/?utm_source=mybridge&utm_medium=blog&utm_campaign=read_more
# импорт необходимых библиотек
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import time
import cv2
import os

path = os.path.join(os.path.abspath(os.curdir), 'pytorch_model.onnx')
# задаем минимальную точнку отсечения вероятности
args_confidence = 0.2

# назначаем метки для классификации
CLASSES = ['cocacola', 'pepsi']

# загружаем ранее подготовленную модель
print("[INFO] loading model...")
net = cv2.dnn.readNetFromONNX(path)

# запускаем видео-стрим, ждём, пока камера включится,
# и начинаем отсчет кадров
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# цикл распознования
while True:
    # считываем кадр из потока,
    # затем заменяем его размер до 400 пикселей
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    # чуть позже нам понадобится ширина и высота, получим их сейчас
    (h, w) = frame.shape[:2]
    # преобразуем кадр в blob с модулем dnn
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (32, 32)), scalefactor=1.0 / 32,
                                 size=(32, 32), mean=(128, 128, 128), swapRB=True)

    # устанавливаем blob как входные данные в нашу нейросеть
    # и затем передаем через net
    net.setInput(blob)
    detections = net.forward()
    confidence = abs(detections[0][0] - detections[0][1])

    # сверяем вероятность с минимальным уровнем
    if confidence > args_confidence:
        class_mark = np.argmax(detections)
        # название наиболее вероятного класса маркируется
        cv2.putText(frame, CLASSES[class_mark], (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (242, 230, 220), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # для остановки программы надо нажать клавишу "q" (quit)
    if key == ord("q"):
        break

    # обновление счетчика кадров
    fps.update()
    
# остановка счетчика кадров
fps.stop()
# закрываем окно программы, прекращая видео поток
cv2.destroyAllWindows()
vs.stop()
