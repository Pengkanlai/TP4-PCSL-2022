import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

objects1 = []
objects2 = []
objects3 = []
objects4 = []
objects5 = []
objects6 = []
objects7 = []
objects8 = []
folder = "C:\data\diagonal linear model"
files = os.listdir(folder)
files= [folder+'\\'+file for file in files]

# model = '/diagonal linear model'
# files = []
# files.append("C:/data"+model+"/alpha10e3")
# files.append("C:/data"+model+"/alpha10e2")
# files.append("C:/data"+model+"/alpha10e1")
# files.append("C:/data"+model+"/alpha10e0")
# files.append("C:/data"+model+"/alpha10e-1")
# files.append("C:/data"+model+"/alpha10e-2")
# files.append("C:/data"+model+"/alpha10e-3")
# files.append("C:/data"+model+"/expbs1")

objects = {}
for iif,file in enumerate(files):
    with (open(file, "rb")) as f:
        objects[iif] = []
        while True:
            try:
                objects[iif].append(pickle.load(f))
            except EOFError:
                break

# plt.figure()
# for key in objects.keys():
#     label = f"alpha = {objects[key][0]['alpha']}"
#     plt.loglog(objects[key][1]['t'], objects[key][1]['dw'], '-',label = label)
# plt.legend()
# plt.show()


dic = {}
for key in objects.keys():
    keyalpha = objects[key][0]['alpha']
    if keyalpha not in dic.keys():
        dic[keyalpha] = {}
        dic[keyalpha]['bs'] = []
        dic[objects[key][0]['alpha']]['T'] = []
    dic[keyalpha]['bs'].append(objects[key][0]['bs'])
    dic[keyalpha]['T'].append(objects[key][1]['t'][-1])

plt.figure()
for keyalpha in dic.keys():
    label = f"alpha = {keyalpha}"
    idx = np.argsort(dic[keyalpha]['bs'])
    dic[keyalpha]['bs'] = np.array(dic[keyalpha]['bs'])
    dic[keyalpha]['T'] = np.array(dic[keyalpha]['T'])
    plt.loglog(dic[keyalpha]['bs'][idx], dic[keyalpha]['T'][idx], '-', label = label)
bsvals = np.linspace(1,128)
plt.loglog(bsvals, 1e3*(bsvals/bsvals[0])**(-0.5), '--', label = "B**(-0.5)")
plt.xlabel('B')
plt.ylabel('total time')
plt.legend()
plt.show()


# with (open(file_name1, "rb")) as f:
#     while True:
#         try:
#             objects1.append(pickle.load(f))
#         except EOFError:
#             break

# with (open(file_name2, "rb")) as f:
#     while True:
#         try:
#             objects2.append(pickle.load(f))
#         except EOFError:
#             break

# with (open(file_name3, "rb")) as f:
#     while True:
#         try:
#             objects3.append(pickle.load(f))
#         except EOFError:
#             break

# with (open(file_name4, "rb")) as f:
#     while True:
#         try:
#             objects4.append(pickle.load(f))
#         except EOFError:
#             break

# with (open(file_name5, "rb")) as f:
#     while True:
#         try:
#             objects5.append(pickle.load(f))
#         except EOFError:
#             break

# with (open(file_name6, "rb")) as f:
#     while True:
#         try:
#             objects6.append(pickle.load(f))
#         except EOFError:
#             break

# with (open(file_name7, "rb")) as f:
#     while True:
#         try:
#             objects7.append(pickle.load(f))
#         except EOFError:
#             break

# with (open(file_name8, "rb")) as f:
#     while True:
#         try:
#             objects8.append(pickle.load(f))
#         except EOFError:
#             break

#nb_steps = len(objects[1]['step'])





# zeros_1 = np.zeros(110)-1
# zeros_2 = np.zeros(50)-1
# zeros_3 = np.zeros(10)-1
# zeros_4 = np.zeros(5)-1

# x1 = objects1[1]['t']
# y1 = objects1[1]['dw']
# #for i in range(110):
#    # y1.append(np.zeros(1)-1)
# #y1 = np.concatenate((y1,zeros_1),axis=0)
# x2 = objects2[1]['t']
# y2 = objects2[1]['dw']
# #for i in range(50):
#     #y2.append(np.zeros(1)-1)
# #y2 = np.concatenate((y2,zeros_2),axis=0)
# x3 = objects3[1]['t']
# y3 = objects3[1]['dw']
# #for i in range(10):
#    # y3.append(np.zeros(1)-1)
# #y3 = np.concatenate((y3,zeros_3),axis=0)
# x4 = objects4[1]['t']
# y4 = objects4[1]['dw']
# #for i in range(5):
#     #y4.append(np.zeros(1)-1)
# #y4 = np.concatenate((y4,zeros_4),axis=0)
# #y5 = objects5[1]['dw']
# #y6 = objects6[1]['dw']
# #y7 = objects7[1]['dw']
# #y8 = objects8[1]['Train_aloss']


# #z = objects[1]['Test_error']

# plt.loglog(x1,y1,'.')
# plt.loglog(x2,y2,'.')
# plt.loglog(x3,y3,'.')
# plt.loglog(x4,y4,'.')
# #plt.plot(x,y5,'.')
# #plt.plot(x,y6,'.')
# #plt.plot(x,y7,'.')
# plt.title('Linear model with different values of alpha')
# #plt.plot(x,y8)
# #plt.xlim(-400,1000)
# #plt.ylim(0,6500)
# plt.xlabel('Time')
# plt.ylabel('Weight variation')
# plt.legend(['alpha=1000','alpha=100','alpha=10','alpha=1','alpha=0.1','alpha=0.01','alpha=0.001'])
# plt.show()