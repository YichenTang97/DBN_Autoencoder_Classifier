import torch
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelEncoder

from modules import DBN

class DBNClassifier(BaseEstimator, ClassifierMixin):
    '''
    Deep Belief Network classifier. 

    The classifier is first pre-trained using the unsupervised data, then unrolled as
    an autoencoder-decoder and further fine-tuned with the unsupervised data. The encoder
    part is then used with an additional linear layer to form a classifier module for performing 
    classification tasks. This classifier module is fine-tuned in a supervised manner.
    

    Parameters
    ----------
    n_hiddens : list of int, default=[500, 100, 20]
        Number of hidden units in each layer of the DBN.

    k : int, default=3
        Number of Gibbs sampling steps in Contrastive Divergence algorithm.

    loss_ae : string or loss function, default='MSELoss'
        Loss function used for pre-training the autoencoder. It must be a string exactly equal to the 
        name of a loss function in torch.nn module (e.g., 'MSELoss', 'CrossEntropyLoss', etc.), as you 
        are importing the loss function. See `torch.nn` for available loss functions.

    loss_clf : string or loss function, default='CrossEntropyLoss'
        Loss function used for supervised fine-tuning of the classifier. It must be a string exactly equal to the 
        name of a loss function in torch.nn module (e.g., 'MSELoss', 'CrossEntropyLoss', etc.), as you 
        are importing the loss function. See `torch.nn` for available loss functions.

    optimizer_ae : string or optimizer object, default='Adam'
        Optimizer used for pre-training the autoencoder. It must be a string exactly equal to the 
        name of an optimizer in torch.optim module (e.g., 'SGD', 'Adam', etc.), as you are importing the 
        optimizer function. See `torch.optim` for available optimizers.

    optimizer_clf : string or optimizer object, default='Adam'
        Optimizer used for supervised fine-tuning of the classifier. It must be a string exactly equal to the 
        name of an optimizer in torch.optim module (e.g., 'SGD', 'Adam', etc.), as you are importing the 
        optimizer function. See `torch.optim` for available optimizers.

    lr_rbm : float, default=1e-5
        Learning rate used for pre-training the DBN using Contrastive Divergence.

    lr_ae : float, default=0.01
        Learning rate used for fine-tuning the autoencoder.

    lr_clf : float, default=0.01
        Learning rate used for supervised fine-tuning of the classifier.

    epochs_rbm : int, default=100
        Number of epochs used for pre-training the DBN using Contrastive Divergence.

    epochs_ae : int, default=50
        Number of epochs used for fine-tuning the autoencoder.

    epochs_clf : int, default=50
        Number of epochs used for supervised fine-tuning of the classifier.

    batch_size_rbm : int, default=50
        Batch size used for pre-training the DBN using Contrastive Divergence.

    batch_size_ae : int, default=50
        Batch size used for fine-tuning the autoencoder.

    batch_size_clf : int, default=50
        Batch size used for supervised fine-tuning of the classifier.

    loss_ae_kwargs : dict, default={}
        Additional keyword arguments to pass to the autoencoder loss function.

    loss_clf_kwargs : dict, default={}
        Additional keyword arguments to pass to the classifier loss function.

    optimizer_ae_kwargs : dict, default={}
        Additional keyword arguments to pass to the autoencoder optimizer.

    optimizer_clf_kwargs : dict, default={}
        Additional keyword arguments to pass to the classifier optimizer.

    random_state : int, default=42
        Seed used by the random number generator.

    use_gpu : bool, default=True
        Whether to use GPU for computation if it's available.

    verbose : bool, default=True
        Whether to print progress messages.

    Attributes
    ----------
    n_features_in_ : int
        Number of input features.

    n_classes_ : int
        Number of target classes.

    le_ : LabelEncoder
        Label encoder object.

    aedbn_ : AutoencoderDBN
        Trained autoencoder-decoder module.

    cdbn_ : ClassifierDBN
        Trained classifier module.

    device_ : torch.device
        Device used for computation (either CPU or GPU).
    '''


    def __init__(self, n_hiddens=[500, 100, 20], k=3, loss_ae='MSELoss', loss_clf='CrossEntropyLoss',
                 optimizer_ae='Adam', optimizer_clf='Adam',
                 lr_rbm=1e-5, lr_ae=0.01, lr_clf=0.01,
                 epochs_rbm=100, epochs_ae=50, epochs_clf=50,
                 batch_size_rbm=50, batch_size_ae=50, batch_size_clf=50,
                 loss_ae_kwargs={}, loss_clf_kwargs={},
                 optimizer_ae_kwargs={}, optimizer_clf_kwargs={}, random_state=42,
                 use_gpu=True, verbose=True):
        self.n_layers = len(n_hiddens)
        self.n_hiddens = n_hiddens
        self.k = k
        self.loss_ae = loss_ae
        self.loss_clf = loss_clf
        self.optimizer_ae = optimizer_ae
        self.optimizer_clf = optimizer_clf
        self.lr_rbm = lr_rbm
        self.lr_ae = lr_ae
        self.lr_clf = lr_clf
        self.epochs_rbm = epochs_rbm
        self.epochs_ae = epochs_ae
        self.epochs_clf = epochs_clf
        self.batch_size_rbm = batch_size_rbm
        self.batch_size_ae = batch_size_ae
        self.batch_size_clf = batch_size_clf
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.loss_ae_kwargs = loss_ae_kwargs
        self.loss_clf_kwargs = loss_clf_kwargs
        self.optimizer_ae_kwargs = optimizer_ae_kwargs
        self.optimizer_clf_kwargs = optimizer_clf_kwargs
        self.random_state = random_state

        if torch.cuda.is_available() and use_gpu==True:
            dev = "cuda:0"
        else:
            dev = "cpu"
        self.device_ = torch.device(dev)

        if not self.random_state is None:
            torch.manual_seed(self.random_state)

    def fit(self, X, y):
        """
        Fit Deep Belief Network classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : DBNClassifier
            Fitted DBNClassifier.

        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.n_classes_ = np.unique(y).shape[0]

        self.le_ = LabelEncoder()
        self.le_.fit(y)

        X = torch.as_tensor(X, dtype=torch.float).to(self.device_)
        y = torch.as_tensor(self.le_.transform(y), dtype=torch.int64).to(self.device_)

        # pretrain DBN
        dbn = DBN(
            self.n_features_in_, 
            self.n_hiddens, 
            lr=self.lr_rbm, 
            epochs=self.epochs_rbm, 
            batch_size=self.batch_size_rbm, 
            k=self.k,
            use_gpu=self.use_gpu, 
            verbose=self.verbose
        )
        dbn.pre_train(X)

        # unroll as autoencoder-decoder and fine tune
        self.aedbn_ = dbn.to_autoencoder(
            loss=self.loss_ae, 
            optimizer=self.optimizer_ae, 
            lr=self.lr_ae,
            epochs=self.epochs_ae, 
            batch_size=self.batch_size_ae, 
            loss_kwargs=self.loss_ae_kwargs, 
            optimizer_kwargs=self.optimizer_ae_kwargs
        )
        self.aedbn_.to(self.device_)
        self.aedbn_.fine_tune(X)

        # use the trained encoder, add an output layer and perform a supervised fine-tune
        self.cdbn_ = self.aedbn_.to_clf(
            n_class=self.n_classes_, 
            loss=self.loss_clf, 
            optimizer=self.optimizer_clf, 
            lr=self.lr_clf, 
            epochs=self.epochs_clf, 
            batch_size=self.batch_size_clf, 
            loss_kwargs=self.loss_clf_kwargs, 
            optimizer_kwargs=self.optimizer_clf_kwargs
        )
        self.cdbn_.to(self.device_)
        self.cdbn_.fine_tune(X, y)

        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        """

        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        X = torch.as_tensor(X).to(self.device_)

        return self.le_.inverse_transform(self.cdbn_.predict(X).cpu().detach().numpy())


class SimpleDBNClassifier(DBNClassifier):
    """
    A simpler version of the Deep Belief Network classifier that allows for easier customization.

    The fixed parameters include: 
    loss_ae='MSELoss', loss_clf='CrossEntropyLoss', optimizer_ae='Adam', optimizer_clf='Adam'

    The classifier assumns the same learning rate, epochs and momentum for fine-tuning the autoencoder
    and classifer module. The same batch size is applied for both pre-training and fine-tuning. 

    Parameters
    ----------
    n_hiddens : list of int, default=[500, 100, 20]
        Number of hidden units in each layer of the DBN.

    lr_pre_train : float, default=1e-5
        Learning rate used for pre-training the DBN using Contrastive Divergence.

    lr_fine_tune : float, default=0.01
        Learning rate used for fine-tuning the autoencoder and classifier.

    epochs_pre_train : int, default=100
        Number of epochs used for pre-training the DBN using Contrastive Divergence.

    epochs_fine_tune : int, default=50
        Number of epochs used for fine-tuning the autoencoder and classifier.

    batch_size : int, default=50
        Batch size used for both pre-training and fine-tuning.

    k : int, default=3
        Number of Gibbs sampling steps in Contrastive Divergence algorithm in the pre-training step.

    random_state : int, default=42
        Seed used by the random number generator.

    use_gpu : bool, default=True
        Whether to use GPU for computation if it's available.

    verbose : bool, default=True
        Whether to print progress messages.

    Attributes
    ----------
    n_features_in_ : int
        Number of input features.

    n_classes_ : int
        Number of target classes.

    le_ : LabelEncoder
        Label encoder object.

    aedbn_ : AutoencoderDBN
        Trained autoencoder-decoder module.

    cdbn_ : ClassifierDBN
        Trained classifier module.

    device_ : torch.device
        Device used for computation (either CPU or GPU).
    """

    def __init__(self, n_hiddens=..., lr_pre_train=1e-5, lr_fine_tune=0.01, 
                 epochs_pre_train=10, epochs_fine_tune=5, batch_size=30, k=3, 
                 random_state=42, use_gpu=True, verbose=True):
        self.lr_pre_train = lr_pre_train
        self.lr_fine_tune = lr_fine_tune
        self.epochs_pre_train = epochs_pre_train
        self.epochs_fine_tune = epochs_fine_tune
        self.batch_size = batch_size

        super().__init__(n_hiddens, k=k, loss_ae='MSELoss', loss_clf='CrossEntropyLoss', 
                         optimizer_ae='Adam', optimizer_clf='Adam',
                         lr_rbm=lr_pre_train, lr_ae=lr_fine_tune, lr_clf=lr_fine_tune,
                         epochs_rbm=epochs_pre_train, epochs_ae=epochs_fine_tune, epochs_clf=epochs_fine_tune,
                         batch_size_rbm=batch_size, batch_size_ae=batch_size, batch_size_clf=batch_size,
                         loss_ae_kwargs={}, loss_clf_kwargs={}, optimizer_ae_kwargs={}, 
                         optimizer_clf_kwargs={}, random_state=random_state,
                         use_gpu=use_gpu, verbose=verbose)
