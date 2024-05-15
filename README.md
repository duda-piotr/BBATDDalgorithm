# BBATDDalgorithm

The package contains the implementation of the methods presented in the paper:
P. Duda, et al. "Accelerating deep neural network learning using data stream methodology.", Information Sciences, 669, (2024): 120575.

It contains four files:

main_class.py - The file contains an implementation of a class enabling streaming learning of the network. The class has a mechanism that allows for network pretraining and subsequntly continue training using different approaches starting from the same point.

experiment_cifar.py - File for testing methods on the CIFAR dataset.
experiment_mnist.py - File for testing methods on the MNIST dataset.

In the above files, users can create a network of any structure (in the keras package) and define parameters of the learning process, i.e.
  - nb: Size of mini-batch;
  - epok: Number of epochs to train the networks. In the streaming apporach, it determines number of iterations equal to: epok*int(len(training_set)/nb);
  - pre: Number of epochs to pretrain the model;
  - lam and alfa: The parameters of CuSum algorithm.

requirments.txt - The list of libraries used in scripts.

The code can use the keras package intended for CPU and GPU, but the version presented here performs all operations sequentially.
