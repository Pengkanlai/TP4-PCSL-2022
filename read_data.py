import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


folder = "C:\data\d_change"
files = os.listdir(folder)
files= [folder+'\\'+file for file in files]

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

#dic = {}
#for key in objects.keys():
    #keyalpha = objects[key][0]['alpha']
    #print(objects[key][0])

    #keyd = objects[key][0]['d']
    #if keyd not in dic.keys():
        #dic[keyd] = {}
        #dic[keyd]['T'] = []
        #dic[objects[key][0]['d']]['Train_loss'] = []
    #dic[keyd]['T'].append(objects[key][1]['t'])
    #dic[keyd]['Train_loss'].append(objects[key][1]['Train_loss'])

plt.figure()
plt.grid()

for key in objects.keys():
    label = f"d = {objects[key][0]['d']}"
    #idx = np.argsort(dic[keyd]['T'])
    objects[key][1]['t'] = np.array(objects[key][1]['t'])
    objects[key][1]['dw'] = np.array(objects[key][1]['dw'])
    plt.loglog(objects[key][1]['t'], objects[key][1]['dw'], '-', label = label)

#bsvals = np.linspace(1,128)
#plt.loglog(bsvals, 1e3*(bsvals/bsvals[0])**(-0.5), '--', label = "B**(-0.5)")


plt.xlabel('time')
plt.ylabel('weight variation')
plt.legend()
plt.show()
