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
file_name1 = "C:/Users/Pengkanlai/Desktop/TP4_PCSL_2022/data/alpha10e3"
file_name2 = "C:/Users/Pengkanlai/Desktop/TP4_PCSL_2022/data/alpha10e2"
file_name3 = "C:/Users/Pengkanlai/Desktop/TP4_PCSL_2022/data/alpha10e1"
file_name4 = "C:/Users/Pengkanlai/Desktop/TP4_PCSL_2022/data/alpha10e0"
file_name5 = "C:/Users/Pengkanlai/Desktop/TP4_PCSL_2022/data/alpha10e-1"
file_name6 = "C:/Users/Pengkanlai/Desktop/TP4_PCSL_2022/data/alpha10e-2"
file_name7 = "C:/Users/Pengkanlai/Desktop/TP4_PCSL_2022/data/alpha10e-3"
file_name8 = "C:/Users/Pengkanlai/Desktop/TP4_PCSL_2022/data/expbs1"

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





zeros_1 = np.zeros(110)-1
zeros_2 = np.zeros(50)-1
zeros_3 = np.zeros(10)-1
zeros_4 = np.zeros(5)-1

x = objects5[1]['t']
y1 = objects1[1]['dw']
for i in range(110):
    y1.append(np.zeros(1)-1)
#y1 = np.concatenate((y1,zeros_1),axis=0)
y2 = objects2[1]['dw']
for i in range(50):
    y2.append(np.zeros(1)-1)
#y2 = np.concatenate((y2,zeros_2),axis=0)
y3 = objects3[1]['dw']
for i in range(10):
    y3.append(np.zeros(1)-1)
#y3 = np.concatenate((y3,zeros_3),axis=0)
y4 = objects4[1]['dw']
for i in range(5):
    y4.append(np.zeros(1)-1)
#y4 = np.concatenate((y4,zeros_4),axis=0)
y5 = objects5[1]['dw']
y6 = objects6[1]['dw']
y7 = objects7[1]['dw']
#y8 = objects8[1]['Train_aloss']


#z = objects[1]['Test_error']

plt.plot(x,y1,'.')
plt.plot(x,y2,'.')
plt.plot(x,y3,'.')
plt.plot(x,y4,'.')
plt.plot(x,y5,'.')
plt.plot(x,y6,'.')
plt.plot(x,y7,'.')
plt.title('Linear model with different values of alpha')
#plt.plot(x,y8)
#plt.xlim(-400,1000)
plt.ylim(0,6500)
plt.xlabel('Time')
plt.ylabel('Weight variation')
plt.legend(['alpha=1000','alpha=100','alpha=10','alpha=1','alpha=0.1','alpha=0.01','alpha=0.001'])
plt.show()