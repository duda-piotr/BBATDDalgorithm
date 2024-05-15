import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

#A file enabling experiments to be performed on the MNIST dataset presented in the paper
#Duda, Piotr, Mateusz Wojtulewicz, and Leszek Rutkowski. "Accelerating deep neural network learning using data stream methodology." Information Sciences 669 (2024): 120575.

#%%
import numpy as np
import matplotlib.pyplot as plt

import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import main_class as ex


#%%
# Prepare data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

chanels=1
num_classes = 10
img_rows, img_cols = 28,28

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, chanels)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, chanels)
input_shape = (img_rows, img_cols, chanels)
    
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test =  tensorflow.keras.utils.to_categorical(y_test, num_classes)

#%%
#Define a network structure
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#%%
#Initilazie experiment
x_train = x_train[:3000]
y_train = y_train[:3000]
x_test = x_test[:1000]
y_test = y_test[:1000]

ex1 = ex.experiment("MNIST", x_train, y_train, x_test, y_test, input_shape, num_classes, model, 1)

nb = 32
epok = 10
pre = 2
lam = 0.1
alfa = 0.01

#%%
ex1.pre_train(nb, pre)
#method, epok, nb, pre_uczenie=2, rep=False, lam=0.05, alfa=0.01
s1 = ex1.start(ex1.MB, epok, nb, pre, lam = lam, alfa = alfa)
s5 = ex1.start(ex1.NLB, epok, nb, pre, lam = lam, alfa = alfa)
s7 = ex1.start(ex1.ENT, epok, nb, pre, lam = lam, alfa = alfa)
s8 = ex1.start(ex1.ELB, epok, nb, pre, lam = lam, alfa = alfa)



#%%
#Creat a plot of a loss function on validation data
mb = np.array(s1[0])[:,0]
nlb = np.array(s5[0])[:,0]
ent = np.array(s7[0])[:,0]
elb = np.array(s8[0])[:,0]
ax = np.arange(0, len(nlb))

plt.plot(ax, mb, label="MB")
plt.plot(ax, nlb, label="NLB")
plt.plot(ax, elb, label="ELB")
plt.plot(ax, ent, label="ENT")
plt.xlabel('Batch number')
plt.ylabel('Loss values')
plt.legend()
plt.savefig(os.path.join(ex1.path,"test_loss.png"))
plt.show()
    
    
#%%
#Creat a plot of an accuracy on validation data
mb = np.array(s1[0])[:,1]
nlb =np.array(s5[0])[:,1]
ent = np.array(s7[0])[:,1]
elb = np.array(s8[0])[:,1]
ax = np.arange(0, len(nlb))

plt.plot(ax, mb, label="MB")
plt.plot(ax, nlb, label="NLB")
plt.plot(ax, elb, label="ELB")
plt.plot(ax, ent, label="ENT")
plt.xlabel('Batch number')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(ex1.path,"test_acc.png"))
plt.show()
    
#%%
#Creat a plot of a loss function on training instances
mb = np.array(s1[1])[:,0]
nlb =np.array(s5[1])[:,0]
ent = np.array(s7[1])[:,0]
elb = np.array(s8[1])[:,0]
ax = np.arange(0, len(nlb))

plt.plot(ax, mb, label="MB")
plt.plot(ax, nlb, label="NLB")
plt.plot(ax, elb, label="ELB")
plt.plot(ax, ent, label="ENT")
plt.xlabel('Batch number')
plt.ylabel('Loss values')
plt.legend()
plt.savefig(os.path.join(ex1.path,"train_loss.png"))
plt.show()
    
    
#%%
#Creat a plot of an accuracy on training instances
mb = np.array(s1[1])[:,1]
nlb =np.array(s5[1])[:,1]
ent = np.array(s7[1])[:,1]
elb = np.array(s8[1])[:,1]
ax = np.arange(0, len(nlb))

plt.plot(ax, mb, label="MB")
plt.plot(ax, nlb, label="NLB")
plt.plot(ax, elb, label="ELB")
plt.plot(ax, ent, label="ENT")
plt.xlabel('Batch number')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(ex1.path,"train_acc.png"))