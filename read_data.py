import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd


folder = "C:\data\T(bs)_l"
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

dic = {}
for key in objects.keys():
    #print(objects[key][0])
    keyalpha = objects[key][0]['alpha']
    if keyalpha not in dic.keys():
        dic[keyalpha] = {}
        dic[keyalpha]['bs'] = []
        dic[objects[key][0]['alpha']]['T'] = []
    dic[keyalpha]['T'].append(objects[key][1]['t'][-1])
    dic[keyalpha]['bs'].append(objects[key][0]['bs'])


#sorted_indices = np.argsort([objects[kk][0]['alpha'] for kk in objects.keys()])

#keys = np.array(list(objects.keys()))[sorted_indices]

sns.set_style('whitegrid')

for keyalpha in dic.keys():
    label = f"alpha = {keyalpha}"
    idx = np.argsort(dic[keyalpha]['bs'])
    dic[keyalpha]['T'] = np.array(dic[keyalpha]['T'])
    dic[keyalpha]['bs'] = np.array(dic[keyalpha]['bs'])
    df = {"bs": dic[keyalpha]['bs'][idx], "T": dic[keyalpha]['T'][idx]}
    df = pd.core.frame.DataFrame(df)
    #ax.set(xscale="log", yscale="log")
    sns.lineplot(x="bs", y="T",ci=50, data = df, label=label)
    #plt.loglog(dic[keyalpha]['bs'][idx], dic[keyalpha]['T'][idx], '-', label = label)

bsvals = np.linspace(1,128)
plt.loglog(bsvals, 1e3*(bsvals/bsvals[0])**(-0.5), '--', label = "B**(-0.5)")

plt.title('Total training time as a funciton of batch size using linear model',fontsize=20)
plt.xlabel('batch size',fontsize=20)
plt.ylabel('training time',fontsize=20)
plt.legend()
plt.show()
