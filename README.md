# DBN_Autoencoder_Classifier
 A pytorch implementation of Deep Belief Network (DBN) with sklearn compatibility for classification. 
 
> **Warning**
> Please note this is not an official implementation, nor has been tested on the datasets used in the original studies. Due to different libraries and hyperparameters used in the implementation (and potentially implementation errors), there might be differences in the performance of this model to the ones as described in the papers. Please always examine the source code, make your own changes if necessary, and describe the actual implementation if you are using this model for an academic study. And please raise an issue if you found any implementation error in my code, thank you!
 
## Introduction

This repository is an implementation and generalisation of the method described in [1], which involves pre-training a DBN using unsupervised data, unrolling it as an autoencoder-decoder, fine-tuning it using unsupervised data, and then fine-tuning the encoder with supervision to use it as a classifier (see Fig.2 in [1]).

The DBN is a stack of Restricted Boltzmann Machines (RBMs), which are trained layer-by-layer during pre-training. The first RBM handeling inputs is a Gaussian-Bernoulli RBM, while the rest are Bernoulli RBMs. Once unrolled into an autoencoder-decoder and fine-tuned, the encoder part is used with an additional output linear layer to perform classification tasks.

I used this model to perform my electroencephalogram (EEG) analyses, hence the default hyperparameters were tuned toward my specific usage. Please always do some hyper parameter tunings before using the model on your dataset.

## Requirements
This model was coded and tested on Python 3.9 with the following libraries and versions (minor differences in versions should not affect the model outcomes):

```Python
numpy >= 1.21.6
scikit-learn >= 1.1.3
torch == 1.13.1+cu116
```

## Examples

See "DBN_example.ipynb".

One can use the full API:

```Python
>>> import numpy as np
>>> from sklearn.datasets import load_digits
>>> from sklearn.model_selection import cross_val_score

>>> from DBNAC import DBNClassifier

>>> X, y = load_digits(return_X_y=True)
>>> print(X.shape)
(1797, 64)

>>> clf = DBNClassifier(n_hiddens=[500, 100, 20], k=3, 
>>>                     loss_ae='MSELoss', loss_clf='CrossEntropyLoss',
>>>                     optimizer_ae='Adam', optimizer_clf='Adam',
>>>                     lr_rbm=1e-5, lr_ae=0.01, lr_clf=0.01,
>>>                     epochs_rbm=100, epochs_ae=50, epochs_clf=50,
>>>                     batch_size_rbm=50, batch_size_ae=50, batch_size_clf=50,
>>>                     loss_ae_kwargs={}, loss_clf_kwargs={},
>>>                     optimizer_ae_kwargs=dict(), optimizer_clf_kwargs=dict(), 
>>>                     random_state=42, use_gpu=True, verbose=False)
>>> scores = cross_val_score(clf, X, y)
>>> print(np.mean(scores))
0.9298777468276075
>>> print(scores)
[0.95       0.91944444 0.94428969 0.95264624 0.88300836]
```

Or use a simplified API with less hyperparameters:

```Python
>>> import numpy as np
>>> from sklearn.datasets import load_digits
>>> from sklearn.model_selection import cross_val_score

>>> from DBNAC import SimpleDBNClassifier

>>> X, y = load_digits(return_X_y=True)
>>> print(X.shape)
(1797, 64)

>>> clf = SimpleDBNClassifier(n_hiddens=[500, 100, 20], lr_pre_train=1e-5, lr_fine_tune=0.01, 
>>>                           epochs_pre_train=100, epochs_fine_tune=50, batch_size=50, k=3, 
>>>                           random_state=42, use_gpu=True, verbose=False)
>>> scores = cross_val_score(clf, X, y)
>>> print(np.mean(scores))
0.9298777468276075
>>> print(scores)
[0.95       0.91944444 0.94428969 0.95264624 0.88300836]
```
 
# Acknowledgements
Special thanks to these repositories for their implementations of RBMs and DBNs which inspired me during my implementation - these are all briliant implementations with different focuses to help you with your own projects:

https://github.com/albertbup/deep-belief-network

https://github.com/DSL-Lab/GRBM

https://github.com/mehulrastogi/Deep-Belief-Network-pytorch

https://github.com/AmanPriyanshu/Deep-Belief-Networks-in-PyTorch

https://github.com/wuaalb/keras_extensions/tree/master/keras_extensions

 
# References
 [1] W. L. Zheng and B. L. Lu, “Investigating Critical Frequency Bands and Channels for EEG-Based Emotion Recognition with Deep Neural Networks,” IEEE Trans. Auton. Ment. Dev., vol. 7, no. 3, pp. 162–175, Sep. 2015, doi: 10.1109/TAMD.2015.2431497.
