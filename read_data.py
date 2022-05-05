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

#print(objects[1])

#nb_steps = len(objects[1]['step'])

x = objects[1]['t']
y = objects[1]['Train_loss']

plt.plot(x,y)
plt.xlabel('Steps')
plt.ylabel('Train Loss')
plt.show()