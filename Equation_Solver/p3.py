import cv2
import numpy as np
from tkinter import *
from tkinter import font as tkFont
from PIL import Image, ImageTk
from tkinter import filedialog

from cv2 import MORPH_OPEN, MORPH_CLOSE
from keras.models import model_from_json


image_display_size = 500, 500

global image_path
global loaded_model


def open_image():

    global image_path
    global loaded_model

    image_path = 0

    image_path = filedialog.askopenfilename(
        filetypes=(('jpg files', '*.jpg'), ('png files', '*.png'), ('jpeg files', '*.jpeg')))

    if (image_path.endswith(".png")):

        json_file = open('model_final.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model_final.h5")

    else:

        json_file = open('create_model_final.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("create_model_final.h5")


    print(image_path)

    if (image_path != None):

        try:

            load_image = Image.open(image_path)

            load_image.thumbnail(image_display_size, Image.ANTIALIAS)

            np_load_image = np.asarray(load_image)
            np_load_image = Image.fromarray(np.uint8(np_load_image))
            render = ImageTk.PhotoImage(np_load_image)
            img = Label(app, image=render)
            img.image = render
            img.place(relx = 0.5, rely = 0.3, anchor = 'center')
            img.config(height=200, width=700)

        except:

            pass



def rect_formation():

    global image_path
    global loaded_model

    img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)  # works by replacing 'cv2.IMREAD_GRAYSCALE' to '0'
    kernel = np.ones((5,5),np.uint8)
    #cv2.imshow("equation",img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    img = cv2.erode(img, kernel, iterations=1)
    cv2.imshow("",img)
    cv2.destroyAllWindows()
    img = cv2.dilate(img, kernel, iterations=1)

    if img is not None:

        img=~img
        img = cv2.morphologyEx(img, MORPH_CLOSE,np.ones((5,5),np.uint8))
        ret,thresh=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3,27),np.uint8))
        ctrs,hierarchy=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnt=sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
        w,h = int(28),int(28)

        train_data=[]

        for c in cnt :

            x,y,w,h = cv2.boundingRect(c)

            cv2.rectangle(img, (x,y),(x+w+10,y+h+10), color=(255,255,255), thickness=2)

            im_crop =thresh[y:y+h+10,x:x+w+10]

            im_resize = cv2.resize(im_crop,(28,28))
            cv2.namedWindow('', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('',900,200)
            cv2.imshow("",img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            im_resize=np.reshape(im_resize,(28,28,1))
            train_data.append(im_resize)

    s = ""
    m = {10:"-", 11:"+", 0:"0", 1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9", 12:"(", 13:")",15:"*"}
    for i in range(len(train_data)):

        train_data[i]=np.array(train_data[i])
        train_data[i]=train_data[i].reshape(1,28,28,1)
        result=np.argmax(loaded_model.predict(train_data[i]), axis=-1)
        for n in m:

            if(result[0]==n):

                s = s + m[n]

    w = s + '=' + (str(eval(s)))

    message_label = Label(app, text=w, bg='white', font=("Courier", 40), width=10, height=5)
    message_label.place(relx = 0.5, rely = 0.8, anchor = 'center')
    message_label.config(width=30, height=5)


app = Tk()
#app.configure(background='orange')
app.title("Encrypt")
w, h = app.winfo_screenwidth(), app.winfo_screenheight()
app.geometry("%dx%d+0+0" % (w, h))

bg = Image.open("C:\\Users\\JAY PATEL\\Desktop\\background.jpg")

bg = bg.resize((1540,795), Image.ANTIALIAS)
my_img = ImageTk.PhotoImage(bg)

canvas1 = Canvas(app, width=1920, height=1080)
canvas1.pack(fill="both", expand=True)
canvas1.create_image(0, 0, image=my_img, anchor="nw")

helv = tkFont.Font(family = 'San serif', size=10, weight=tkFont.BOLD)

image_button = Button(app, text="Solve Equation", font=helv, bg='blue', fg='white', command=rect_formation)
image_button.place(x=793, y=400)

equation_button = Button(app, text="Open Image", font=helv, bg='blue', fg='white', command=open_image)
equation_button.place(x=663, y=400)

label1 = Label(app, text= 'Image Display', bd = 0, font = ('',10))
label1.place(relx = 0.5, rely = 0.3, anchor = 'center')
label1.config(width=90, height=15)

app.mainloop()