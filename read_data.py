import pickle
import matplotlib.pyplot as plt
import numpy as np

objects = []
file_name = "C:/Users/Pengkanlai/Desktop/TP4_PCSL_2022/test"

with (open(file_name, "rb")) as f:
    while True:
        try:
            objects.append(pickle.load(f))
        except EOFError:
            break

print(objects)

#x = np.linspace(1000,10000,10)
#print(x)
#y = objects[0][1]['Train_loss']

#plt.plot(x,y)
#plt.xlabel('Steps')
#plt.ylabel('Train Loss')
#plt.show()