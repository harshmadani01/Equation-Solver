import numpy as np
import cv2
import os
import pandas as pd

def folder_images(folder):
    final = []

    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),cv2.IMREAD_GRAYSCALE)
        img = ~img
        if img is not None:
            ret,thresh=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
            ctrs,hierarchy=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            cnt=sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
            mxm=0
            for c in cnt:
                x,y,w,h=cv2.boundingRect(c)
                mxm=max(w*h,mxm)
                if mxm==w*h:
                    x_max, y_max, w_max, h_max = x, y, w, h
            im_reshape = thresh[y_max:y_max+h_max+10, x_max:x_max+w_max+10]
            im_resize = cv2.resize(im_reshape,(28,28))
            im_resize = np.reshape(im_resize,(784,1))
            final.append(im_resize)
    return final

data=[]


data=folder_images('C:\\Users\\JAY PATEL\\Downloads\\archive (3)\\extracted_images\\-')

for i in range(0,len(data)):
    data[i]=np.append(data[i],['10'])

print(len(data))

data11=folder_images('C:\\Users\\JAY PATEL\\Downloads\\archive (3)\\extracted_images\\+')

for i in range(0,len(data11)):
    data11[i]=np.append(data11[i],['11'])
data=np.concatenate((data,data11))
print(len(data))

data0=folder_images('C:\\Users\\JAY PATEL\\Downloads\\archive (3)\\extracted_images\\0')
for i in range(0,len(data0)):
    data0[i]=np.append(data0[i],['0'])
data=np.concatenate((data,data0))
print(len(data))

data1=folder_images('C:\\Users\\JAY PATEL\\Downloads\\archive (3)\\extracted_images\\1')

for i in range(0,len(data1)):
    data1[i]=np.append(data1[i],['1'])
data=np.concatenate((data,data1))
print(len(data))

data2=folder_images('C:\\Users\\JAY PATEL\\Downloads\\archive (3)\\extracted_images\\2')

for i in range(0,len(data2)):
    data2[i]=np.append(data2[i],['2'])
data=np.concatenate((data,data2))
print(len(data))

data3=folder_images('C:\\Users\\JAY PATEL\\Downloads\\archive (3)\\extracted_images\\3')

for i in range(0,len(data3)):
    data3[i]=np.append(data3[i],['3'])
data=np.concatenate((data,data3))
print(len(data))

data4=folder_images('C:\\Users\\JAY PATEL\\Downloads\\archive (3)\\extracted_images\\4')

for i in range(0,len(data4)):
    data4[i]=np.append(data4[i],['4'])
data=np.concatenate((data,data4))
print(len(data))

data5=folder_images('C:\\Users\\JAY PATEL\\Downloads\\archive (3)\\extracted_images\\5')

for i in range(0,len(data5)):
    data5[i]=np.append(data5[i],['5'])
data=np.concatenate((data,data5))
print(len(data))

data6=folder_images('C:\\Users\\JAY PATEL\\Downloads\\archive (3)\\extracted_images\\6')

for i in range(0,len(data6)):
    data6[i]=np.append(data6[i],['6'])
data=np.concatenate((data,data6))
print(len(data))

data7=folder_images('C:\\Users\\JAY PATEL\\Downloads\\archive (3)\\extracted_images\\7')

for i in range(0,len(data7)):
    data7[i]=np.append(data7[i],['7'])
data=np.concatenate((data,data7))
print(len(data))

data8=folder_images('C:\\Users\\JAY PATEL\\Downloads\\archive (3)\\extracted_images\\8')

for i in range(0,len(data8)):
    data8[i]=np.append(data8[i],['8'])
data=np.concatenate((data,data8))
print(len(data))

data9=folder_images('C:\\Users\\JAY PATEL\\Downloads\\archive (3)\\extracted_images\\9')

for i in range(0,len(data9)):
    data9[i]=np.append(data9[i],['9'])
data=np.concatenate((data,data9))
print(len(data))

data12=folder_images('C:\\Users\\JAY PATEL\\Downloads\\archive (3)\\extracted_images\\(')

for i in range(0,len(data12)):
    data12[i]=np.append(data12[i],['12'])
data=np.concatenate((data,data12))
print(len(data))

data13=folder_images('C:\\Users\\JAY PATEL\\Downloads\\archive (3)\\extracted_images\\)')

for i in range(0,len(data13)):
    data13[i]=np.append(data13[i],['13'])
data=np.concatenate((data,data13))
print(len(data))

data14=folder_images('C:\\Users\\JAY PATEL\\Downloads\\archive (3)\\extracted_images\\div')

for i in range(0,len(data14)):
    data14[i]=np.append(data14[i],['14'])
data=np.concatenate((data,data14))
print(len(data))

data15=folder_images('C:\\Users\\JAY PATEL\\Downloads\\archive (3)\\extracted_images\\times')

for i in range(0,len(data15)):
    data15[i]=np.append(data15[i],['15'])
data=np.concatenate((data,data15))
print(len(data))

data16=folder_images('C:\\Users\\JAY PATEL\\Downloads\\archive (3)\\extracted_images\\sqrt')

for i in range(0,len(data16)):
    data16[i]=np.append(data16[i],['16'])
data=np.concatenate((data,data16))
print(len(data))

data17=folder_images('C:\\Users\\JAY PATEL\\Downloads\\archive (3)\\extracted_images\\log')

for i in range(0,len(data17)):
    data17[i]=np.append(data17[i],['17'])
data=np.concatenate((data,data17))
print(len(data))

df=pd.DataFrame(data,index=None)
df.to_csv('C:\\Users\\JAY PATEL\\Desktop\\train.csv',index=False)