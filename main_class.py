#The package contains the implementation of the methods presented in the article:
#Duda, Piotr, Mateusz Wojtulewicz, and Leszek Rutkowski. "Accelerating deep neural network learning using data stream methodology." Information Sciences 669 (2024): 120575.

import os
import copy
import json 
import datetime
import numpy as np
import tensorflow

from math import  tanh
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten

#os.environ["CUDA_VISIBLE_DEVICES"]="0"

class experiment:
    def __init__(self, name, data_x, data_y, test_x, test_y, input_shape, 
                 num_classes, model, after_every=1):
        """
        A class enable to start the experiments presented in paper [1]
        P. Duda, et.al., "Accelerating deep neural network learning using data stream methodology." Information Sciences 669 (2024): 120575.

        Parameters
        ----------
        name : str
            Name for experiment (given by the user).
        data_x : ndarray
            Input data.
        data_y : ndarray
            Target data.
        test_x : ndarray
            Input test data.
        test_y : ndarray
            Input test data.
        input_shape : tuple
            Shape of network's input.
        num_classes : int
            Number of classes.
        model : keras.engine.sequential.Sequential
            Compiled model of neural network created in Keras library.
        after_every : int, optional
            Number of proacessed baches to check its performance on validation data. The default is 1.

        Returns
        -------
        None.

        """
        self.set_name = name

        self.x_train = data_x
        self.y_train = data_y

        self.x_test = test_x
        self.y_test = test_y

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.ae = after_every

        self.model = model

        self.history = []
        self.history_t = []

        self.vec_mass = np.tile(1/len(self.x_train),len(self.x_train))
        self.vec_count = np.ones(len(self.x_train))*2


    
    def pre_train(self, nb, pre_uczenie):
        """
        Initial (epoch-based) pretraining of the model

        Parameters
        ----------
        nb : int
            batch size.
        pre_uczenie : int
            Number of epochs to pretrain the model.

        Returns
        -------
        None.

        """

        #Create lists to statistics during progress
        self.history = []
        self.history_t = []

        #Initialise pod-s values
        self.vec_mass = np.tile(1/len(self.x_train),len(self.x_train))
        self.vec_count = np.ones(len(self.x_train))*pre_uczenie

        #Train the model
        for epochs_nr in range (0, pre_uczenie):
            for batch_nr in range(0,int(len(self.x_train)/nb)):
                data =  self.x_train[batch_nr*nb:(batch_nr+1)*nb]
                y_data = self.y_train[batch_nr*nb:(batch_nr+1)*nb]
                self.model.train_on_batch(data, y_data)
                #Save statistic after each fixed step
                if (batch_nr +1) % self.ae == 0: 
                    self.history.append(self.model.test_on_batch(self.x_test, self.y_test))
                    self.history_t.append(self.model.test_on_batch(data, y_data))

        #Start point (used in directory name)
        x = datetime.datetime.now()

        #Create a directory for this experiment and save pretained model
        self.path= self.set_name+x.strftime("_%Y-%m-%d-%H-%M-%S")
        os.mkdir(self.path)           
        self.model.save(os.path.join(self.path,"model"))

    #Function allowing execution of epoch-based approach
    def MB(self, nb, batch_nr, model, vec_mass, vec_count, history, rep=False):
        data =  self.x_train[batch_nr*nb:(batch_nr+1)*nb]
        y_data = self.y_train[batch_nr*nb:(batch_nr+1)*nb]
        model.train_on_batch(data, y_data)
        return np.arange(batch_nr*nb,(batch_nr+1)*nb)

    #Function allowing execution of streaming approach with uchenged uniform pods.
    def SMB(self, nb, batch_nr, model, vec_mass, vec_count, history, rep=False):
        indeksy = np.random.choice(range(len(self.x_train)), replace = rep, size=nb, p=vec_mass)
        data =  self.x_train[indeksy, : ]
        y_data = self.y_train[indeksy, : ]
        model.train_on_batch(data, y_data)
        vec_count[indeksy] += 1
        return indeksy

    #Function allowing execution of streaming approach with Normalalized Loss Based approach (see [31]).
    def NLB(self, nb, batch_nr, model, vec_mass, vec_count, history, rep=False):
        indeksy = np.random.choice(range(len(self.x_train)), replace = rep, size=nb, p=vec_mass)
        data =  self.x_train[indeksy, : ]
        y_data = self.y_train[indeksy, : ]
        for i in range(nb):
            blad = model.train_on_batch(data[[i], :], y_data[[i],:])[0]
            vec_count[indeksy[i]] += 1
            vec_mass[indeksy[i]] = tanh(blad) / vec_count[indeksy[i]] 
        
        vec_mass[indeksy] /= np.sum(vec_mass[indeksy])
        vec_mass = np.multiply(vec_mass, 1/np.sum(vec_mass))
        
        self.cusum += np.max([0, history[-2][0] - history[-1][0] - self.alfa])
        if self.cusum > self.lam:
          self.drift_count += 1
          self.drift_when.append(batch_nr)
          vec_mass = np.tile(1/len(self.x_train),len(self.x_train))
          self.cusum = 0
          self.ile += 1
        return indeksy
    
    
    #Function allowing execution of streaming approach with Entropy Based approach (equation (6)]).
    def ENT(self, nb, batch_nr, model, vec_mass, vec_count, history, rep=False):
        indeksy = np.random.choice(range(len(self.x_train)), replace = rep, size=nb, p=vec_mass)
        data =  self.x_train[indeksy, : ]
        y_data = self.y_train[indeksy, : ]

        for i in range(nb):
            model.train_on_batch(data[[i], :], y_data[[i],:])
            blad = model.predict(data[[i], :])
            blad = -np.sum(blad*np.log2(blad, where=(blad!=0)))
            vec_count[indeksy[i]] += 1
            vec_mass[indeksy[i]] = blad

        vec_mass[indeksy] /= np.sum(vec_mass[indeksy])
        vec_mass = np.multiply(vec_mass, 1/np.sum(vec_mass))

        self.cusum += np.max([0, history[-2][0] - history[-1][0] - self.alfa])
        if self.cusum > self.lam:
            vec_mass = np.tile(1/len(self.x_train),len(self.x_train))
            self.cusum = 0
            self.ile += 1
        return indeksy

    #Function allowing execution of streaming approach with Entropy Based approach (equation (5)]).
    def ELB(self, nb, batch_nr, model, vec_mass, vec_count, history, rep=False):
        indeksy = np.random.choice(range(len(self.x_train)), replace = rep, size=nb, p=vec_mass)
        data =  self.x_train[indeksy, : ]
        y_data = self.y_train[indeksy, : ]
        for i in range(nb):
            blad = model.train_on_batch(data[[i], :], y_data[[i],:])[0]
            vec_count[indeksy[i]] += 1
            vec_mass[indeksy[i]] = np.exp(blad)
            
        vec_mass[indeksy] /= np.sum(vec_mass[indeksy])
        vec_mass = np.multiply(vec_mass, 1/np.sum(vec_mass))
        
        self.cusum += np.max([0, history[-2][0] - history[-1][0] - self.alfa])
        if self.cusum > self.lam:
            vec_mass = np.tile(1/len(self.x_train),len(self.x_train))
            self.cusum = 0
            self.ile += 1
        return indeksy


    def start(self, method, epok, nb, pre_uczenie=2, rep=False, lam=0.05, alfa=0.01):
        """
        The main method, enable to start the experiments in a chosen scenario
   
        Parameters
        ----------
        method : one of an inner methods
            Metchod of training. Aviable self.MB, self.SMB, self.NLB, self.ENT or self.ELB
        epok : int
            Number of epochs to train the networks. In the streaming apporach, 
            it determines number of iterations equal epok*int(len(self.x_train)/nb)
        nb : int
            Mini-batch size.
        pre_uczenie : int, optional
            Number of epochs to pretrain the model. The default is 2.
        rep : bool, optional
            Sapling with (True) or without (Flase) replacemnet. The default is False.
        lam : flaot, optional
            Parameter of the drift detenctor (see description under formula (11)). The default is 0.05.
        alfa : float, optional
            Parameter of the drift detenctor (see formula (11)). The default is 0.01.

        Returns
        -------
        history : list
            Table of metrics computed on validation data
        history_t : list
            Table of metrics computed on training data (not on a whole training dataset).
        vec_mass : ndarray
            Final values of pods.
        vec_count : ndarray
            Final number of how many times a data element was used to train the network.
        TYPE: int
            Number of detected drifts.

        """
        self.lam = lam
        self.cusum = 0.0
        self.alfa = alfa
        self.ile = 0

        #Create a new model and set its weights to formerly pre-trained one.
        model = tensorflow.keras.models.clone_model(self.model)
        model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                      optimizer=tensorflow.keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        for i in range(len(self.model.layers)):
            if self.model.layers[i].__class__.__name__ in ["Conv2D", "Dense"]:
                model.layers[i].set_weights(self.model.layers[i].get_weights())

        #Use history evaluation data of pretrained model
        history = copy.deepcopy(self.history)
        history_t = copy.deepcopy(self.history_t)

        #Reinitialized pods values
        vec_mass = np.tile(1/len(self.x_train),len(self.x_train))
        vec_count = np.zeros(len(self.x_train))
        self.drift_count = 0
        self.drift_when = []


        #Start training after pretrained phase
        for epochs_nr in range (pre_uczenie, epok):
            
            for batch_nr in range(0,int(len(self.x_train)/nb)):
                vec_mass = np.multiply(vec_mass, 1/np.sum(vec_mass))

                indeksy = method(nb, batch_nr, model, vec_mass, vec_count, history)
                
                if (batch_nr+1) %self.ae == 0:
                    history.append(model.test_on_batch(self.x_test, self.y_test))
                    history_t.append(model.test_on_batch(self.x_train[indeksy, : ], self.y_train[indeksy, : ]))
                K.clear_session()
            print("Type: {0}, Epok: {1}:\t {2}".format(method.__name__, epochs_nr, history[-1][0]))

        #After training save results to file
        json_data = {
        'loss': np.array(history_t)[:,0].tolist()[-1],
        'val_loss': np.array(history)[:,0].tolist()[-1],
        'acc' : np.array(history_t)[:,1].tolist()[-1],
        'val_acc':  np.array(history)[:,1].tolist()[-1],
        'vec_count': self.ile,
        'epok': epok,
        'nb' :nb,
        'pre_uczenie': pre_uczenie,
        'rep': rep,
        'dataset_name': self.set_name,
        'drift_count': self.drift_count,
        'drift_when': self.drift_when,
        'lambda': self.lam,
        'alfa': self.alfa
        }

        #End point (used in file name)
        x = datetime.datetime.now()
        pom = x.strftime("_%Y-%m-%d-%H-%M-%S")

        nazwa = method.__name__+pom+'.json'
        with open(os.path.join(self.path,nazwa), 'w') as fp:
            json.dump(json_data, fp, sort_keys=True, indent=4)
        
        return history, history_t, vec_mass, vec_count, self.ile