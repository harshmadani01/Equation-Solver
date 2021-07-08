import cv2 as cv
import os
import matplotlib.pyplot as plt
import seaborn as sn
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from PIL import Image

def mnsit_data():

    path = 'C:\\Users\\JAY PATEL\\Downloads\\mnist.npz'

    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test,y_test) = mnist.load_data(path=path)

    return (x_train, y_train, x_test, y_test)

def tm(x_train, y_train, x_test, y_test):

    class callback(tf.keras.callbacks.Callback):

        def on_epoch_end(self, epoch, logs={}):

            print(logs)

            if(logs.get('accuracy') > 0.97):

                self.model.stop_training = True

    callbacks = callback()

    x_train, x_test = x_train/255.0, x_test/255.0  #Data Normalization

    model = tf.keras.models.Sequential([

        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)

    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())

    history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
    print(history.epoch, history.history['accuracy'][-1])
    #fig = plt.figure(figsize=(10,5))



    return model

def predict(model, img):

    imgs = np.array([img])
    res = model.predict(imgs)
    index = np.argmax(res)
    return str(index)

Inference = False

def Event(event, x,y, flags, parameters):

    global Inference
    if event == cv.EVENT_LBUTTONDOWN:
        Inference = not Inference

threshold = 100
def on_threshold(x):
    global threshold
    threshold = x

def start_cv(model):
    global threshold
    cap = cv.imread('C:/Users/JAY PATEL/Desktop/image3.JPG',cv.IMREAD_GRAYSCALE)
    cv.imshow("equation", cap)
    cv.waitKey(0)
    cv.destroyAllWindows()
    #cv.setMouseCallback('background', Event)
    #cv.createTrackbar('threshold', 'background', 150, 255, on_threshold)
    background = np.zeros((480, 640), np.uint8)
    frameCount=0

    if cap is not None:

        img = ~cap
        ret, thresh = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        cv.imshow("Binary Image", thresh)
        cv.waitKey(0)
        ctrs, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnt = sorted(ctrs, key=lambda ctr: cv.boundingRect(ctr)[0])
        w, h = int(28), int(28)

        train_data = []

        for c in cnt:
            x, y, w, h = cv.boundingRect(c)

            cv.rectangle(img, (x, y), (x + w + 20, y + h + 20), color=(255, 255, 0), thickness=2)

            im_crop = thresh[y:y + h + 10, x:x + w + 10]

            im_resize = cv.resize(im_crop, (28, 28))
            cv.imshow("", img)
            cv.waitKey(0)
            cv.destroyAllWindows()


            im_resize = np.reshape(im_resize, (28, 28, 1))

            res = predict(model, im_resize)

            train_data.append(img)

    """while True:
        image = cv.imread('C:/Users/JAY PATEL/Desktop/Output.JPG')
        x, y = 10, 20
        cv.putText(image, "", (x + 10, y + 10), cv.FONT_HERSHEY_SIMPLEX, 1, 255)
        cv.imshow('output', image)
        if cv.waitKey(0) & 0xFF == ord('q'):
            break"""


def main():

    model = None

    (x_train, y_train, x_test, y_test) = mnsit_data()
    model = tm(x_train, y_train, x_test, y_test)
    model.save('C:\\Users\\JAY PATEL\\Desktop\\mnist.sav')

    start_cv(model)

if __name__=='__main__':

    main()
