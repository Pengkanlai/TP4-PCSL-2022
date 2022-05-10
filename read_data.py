import pickle
import matplotlib.pyplot as plt
import numpy as np

objects1 = []
objects2 = []
objects3 = []
objects4 = []
objects5 = []
objects6 = []
objects7 = []
objects8 = []
file_name1 = "C:/Users/Pengkanlai/Desktop/TP4_PCSL_2022/expbs128"
file_name2 = "C:/Users/Pengkanlai/Desktop/TP4_PCSL_2022/expbs64"
file_name3 = "C:/Users/Pengkanlai/Desktop/TP4_PCSL_2022/expbs32"
file_name4 = "C:/Users/Pengkanlai/Desktop/TP4_PCSL_2022/expbs16"
file_name5 = "C:/Users/Pengkanlai/Desktop/TP4_PCSL_2022/expbs8"
file_name6 = "C:/Users/Pengkanlai/Desktop/TP4_PCSL_2022/expbs4"
file_name7 = "C:/Users/Pengkanlai/Desktop/TP4_PCSL_2022/expbs2"
file_name8 = "C:/Users/Pengkanlai/Desktop/TP4_PCSL_2022/expbs1"

with (open(file_name1, "rb")) as f:
    while True:
        try:
            objects1.append(pickle.load(f))
        except EOFError:
            break

with (open(file_name2, "rb")) as f:
    while True:
        try:
            objects2.append(pickle.load(f))
        except EOFError:
            break

with (open(file_name3, "rb")) as f:
    while True:
        try:
            objects3.append(pickle.load(f))
        except EOFError:
            break

with (open(file_name4, "rb")) as f:
    while True:
        try:
            objects4.append(pickle.load(f))
        except EOFError:
            break

with (open(file_name5, "rb")) as f:
    while True:
        try:
            objects5.append(pickle.load(f))
        except EOFError:
            break

with (open(file_name6, "rb")) as f:
    while True:
        try:
            objects6.append(pickle.load(f))
        except EOFError:
            break

with (open(file_name7, "rb")) as f:
    while True:
        try:
            objects7.append(pickle.load(f))
        except EOFError:
            break

with (open(file_name8, "rb")) as f:
    while True:
        try:
            objects8.append(pickle.load(f))
        except EOFError:
            break

#nb_steps = len(objects[1]['step'])


x = objects1[1]['t']
y1 = objects1[1]['Train_loss']
y2 = objects2[1]['Train_loss']
y3 = objects3[1]['Train_loss']
y4 = objects4[1]['Train_loss']
y5 = objects5[1]['Train_loss']
y6 = objects6[1]['Train_loss']
y7 = objects7[1]['Train_loss']
y8 = objects8[1]['Train_loss']

#z = objects[1]['Test_error']

plt.plot(x,y1)
plt.plot(x,y2)
plt.plot(x,y3)
plt.plot(x,y4)
plt.plot(x,y5)
plt.plot(x,y6)
plt.plot(x,y7)
plt.plot(x,y8)
plt.xlim(-400,5000)
plt.xlabel('Time')
plt.ylabel('Train loss')
plt.legend(['bs=128','bs=64','bs=32','bs=16','bs=8','bs=4','bs=2','bs=1'])
plt.show()