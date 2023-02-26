import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from importlib import import_module

from RBM import RBM, GBRBM

class DBN(nn.Module):
    '''
    Deep Belief Network (DBN) implemented using PyTorch. 
    The DBN is the stack of an input GBRBM layer with multiple hidden RBM layers.

    Parameters
    ----------
    n_visible : int
        The number of input units.
    
    n_hiddens : list of int
        A list of integers representing the number of hidden units for the Restricted Boltzmann Machines (RBMs). 
        The length of the list determines the number of RBMs used in the network.
    
    lr : float, optional (default=1e-5)
        The learning rate used for training the RBMs.
    
    epochs : int, optional (default=100)
        The number of epochs used to train each RBM.
    
    batch_size : int, optional (default=50)
        The number of samples used in each training batch.
    
    k : int, optional (default=3)
        The number of contrastive divergence steps used to train each RBM.
    
    use_gpu : bool, optional (default=True)
        A boolean flag indicating whether or not to use GPU acceleration.
    
    verbose : bool, optional (default=True)
        A boolean flag indicating whether or not to print information about the training progress.
    '''

    def __init__(self, n_visible, n_hiddens, lr=1e-5, epochs=100, batch_size=50, k=3,
                 use_gpu=True, verbose=True):
        super(DBN,self).__init__()

        self.n_layers = len(n_hiddens)
        self.n_visible = n_visible
        self.n_hiddens = n_hiddens
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.k = k

        self.rbm_layers_ = []
        for i in range(self.n_layers):
            if i == 0:
                n_in = n_visible
                rbm = GBRBM(
                    n_in, 
                    n_hiddens[0], 
                    lr=lr, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    k=k, 
                    use_gpu=use_gpu, 
                    verbose=verbose
                )
            else:
                n_in = n_hiddens[i-1]
                rbm = RBM(n_in, 
                    n_hiddens[i], 
                    lr=lr, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    k=k, 
                    use_gpu=use_gpu, 
                    verbose=verbose
                )
            self.rbm_layers_.append(rbm)
        
    def forward(self, X):
        '''
        A single forward pass to obtain .

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        p_h : tensor
            The probabilities of the hidden units on the top RBM.
        
        h : tensor
            The activations of the hidden units on the top RBM.
        '''
        h = torch.as_tensor(X, dtype=torch.float)
        for rbm in self.rbm_layers_:
            h = h.view((h.shape[0], -1)) # flatten
            p_h, h = rbm.v_to_h(h)
        return p_h, h

    def pre_train(self, X):
        '''
        Pre-train the DBN one RBM at a time.

        Parameters
        ----------
        X : PyTorch tensor, shape (n_samples, n_features)
            The input data to pre-train the DBN.

        Returns
        -------
        None

        '''
        y = torch.zeros(X.shape[0])

        # train RBMs layer by layer
        for rbm in self.rbm_layers_:
            dataset = TensorDataset(X, y)
            rbm.train(dataset)

            # forward to next rbm
            X = X.view((X.shape[0], -1)) # flatten
            _, X = rbm.forward(X)

    def to_autoencoder(self, loss='MSELoss', optimizer='Adam', lr=0.01,
                       epochs=50, batch_size=50, loss_kwargs={}, optimizer_kwargs=dict()):
        """
        Unroll the DBN into an autoencoder-decoder.

        Parameters:
        -----------
        loss : str or callable, optional (default='MSELoss')
            The loss function used for fine-tuning the autoencoder. It must be a string exactly equal to the 
            name of a loss function in torch.nn module (e.g., 'MSELoss', 'CrossEntropyLoss', etc.), as you
            are importing the loss function.
        
        optimizer : str or torch.optim.Optimizer, optional (default='Adam')
            The optimizer used for fine-tuning the autoencoder. It must be a string exactly equal to the name
            of an optimizer in torch.optim module (e.g., 'SGD', 'Adam', etc.), as you are importing the 
            optimizer function.
        
        lr : float, optional (default=0.01)
            The learning rate used for fine-tuning the autoencoder.
        
        epochs : int, optional (default=50)
            The number of epochs used for fine-tuning the autoencoder.
        
        batch_size : int, optional (default=50)
            The batch size used for fine-tuning the autoencoder.
        
        loss_kwargs : dict, optional (default={})
            Additional keyword arguments to be passed to the loss function.
        
        optimizer_kwargs : dict, optional (default={})
            Additional keyword arguments to be passed to the optimizer.

        Returns:
        --------
        None

        See Also:
        --------
        modules.AEDBN
        """
        return AEDBN(self, loss=loss, optimizer=optimizer, lr=lr, epochs=epochs, batch_size=batch_size,
                     loss_kwargs=loss_kwargs, optimizer_kwargs=optimizer_kwargs, verbose=self.verbose)


class AEDBN(nn.Module):
    """
    A class that constructs an autoencoder based on the DBN model.

    Parameters:
    ----------
    dbn : DBN
        A trained DBN object to construct an autoencoder from.
    
    loss : str, optional (default='MSELoss')
        The name of the loss function used to fine-tune the autoencoder. It must be a string exactly equal to 
        the name of a loss function in torch.nn module (e.g., 'MSELoss', 'CrossEntropyLoss', etc.), as you
        are importing the loss function.
    
    optimizer : str, optional (default='Adam')
        The name of the optimizer used to fine-tune the autoencoder. It must be a string exactly equal to the 
        name of an optimizer in torch.optim module (e.g., 'SGD', 'Adam', etc.), as you are importing the 
        optimizer function.
    
    lr : float, optional (default=0.01)
        The learning rate used by the optimizer during fine-tuning.
    
    epochs : int, optional (default=50)
        The number of epochs used during fine-tuning.
    
    batch_size : int, optional (default=50)
        The batch size used during fine-tuning.
    
    loss_kwargs : dict, optional (default={})
        A dictionary of keyword arguments passed to the loss function.
    
    optimizer_kwargs : dict, optional (default={})
        A dictionary of keyword arguments passed to the optimizer.
    
    verbose : bool, optional (default=True)
        Whether to print the loss during fine-tuning.
    """
    def __init__(self, dbn, loss='MSELoss', optimizer='Adam', lr=0.01, epochs=50, batch_size=50, 
                 loss_kwargs={}, optimizer_kwargs=dict(), verbose=True):
        super(AEDBN,self).__init__()

        self.dbn = dbn
        self.loss = loss
        self.optimizer = optimizer
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.loss_kwargs = loss_kwargs
        self.optimizer_kwargs = optimizer_kwargs

        self.construct_autoencoder()

    def construct_autoencoder(self):
        '''
        Constructs an autoencoder based on the pre-trained Deep Belief Network.

        The autoencoder is constructed by unrolling the layers of the DBN and using the learned weights and biases of each
        Restricted Boltzmann Machine (RBM) to initialize the weights and biases of the corresponding encoder and decoder
        layers. The encoder and decoder layers are then assembled into a single PyTorch module.

        Returns:
        --------
        None

        '''
        # unroll as an anto encoder/decoder
        n_in = self.dbn.n_visible

        # encoder part
        modules = []
        for n_hidden, rbm in zip(self.dbn.n_hiddens, self.dbn.rbm_layers_):
            layer = nn.Linear(n_in, n_hidden)
            layer.weight, layer.bias = nn.Parameter(rbm.W.t()), nn.Parameter(rbm.hb)
            modules.append(layer)
            modules.append(nn.Sigmoid())
            n_in = n_hidden
        self.encoder_ = nn.Sequential(*modules)

        # decoder part
        modules = []
        for i, n_hidden in enumerate(reversed(self.dbn.n_hiddens)):
            if i > 0:
                layer = nn.Linear(n_in, n_hidden)
                layer.weight, layer.bias = nn.Parameter(self.dbn.rbm_layers_[-i].W), nn.Parameter(self.dbn.rbm_layers_[-i].vb)
                modules.append(layer)
                modules.append(nn.Sigmoid())
                n_in = n_hidden
        layer = nn.Linear(n_hidden, self.dbn.n_visible)
        layer.weight, layer.bias = nn.Parameter(self.dbn.rbm_layers_[0].W), nn.Parameter(self.dbn.rbm_layers_[0].vb)
        modules.append(layer) # final output layer
        self.decoder_ = nn.Sequential(*modules)
        
    def forward(self, X):
        """
        Performs a forward pass through the autoencoder.

        Parameters:
        ----------
        X : PyTorch tensor, shape (n_samples, n_features)
            The input data.

        Returns:
        -------
        decoded_X : tensor, shape (n_samples, n_features)
            The reconstructed input data.
        """
        X = torch.as_tensor(X, dtype=torch.float)
        enc = self.encoder_(X)
        return self.decoder_(enc)

    def fine_tune(self, X, y=None):
        """
        Fine-tunes the AEDBN model.

        Parameters:
        ----------
        X : PyTorch tensor, shape (n_samples, n_features)
            The input data.

        y : PyTorch tensor, shape (n_samples, n_features), optional (default=None)
            The target output data. If None, use the input data as target (i.e. unsupervised learning).

        Returns:
        -------
        None

        Notes:
        ------
        A use case for the y parameter is the noisy version of input X, to allow the autoencoder and decoder to 
        endure noise as well as perform denoising tasks.
        """
        X = torch.as_tensor(X, dtype=torch.float)
        if y is None:
            y = X.detach().clone()

        # create dataset and data loader
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size)

        # set loss function and optimizer
        loss_fn = getattr(import_module('torch.nn'), self.loss)(**self.loss_kwargs)
        optimizer = getattr(import_module('torch.optim'), self.optimizer)
        optimizer = optimizer(self.parameters(), lr=self.lr, **self.optimizer_kwargs)

        # switch to training mode
        self.train(True)

        # train the model
        for ep in range(self.epochs):
            running_loss = 0.
            
            for i, (batch, y_batch) in enumerate(loader):
                optimizer.zero_grad()
                outputs = self.forward(batch)
                loss = loss_fn(outputs, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(loader)

            if self.verbose:
                print(f'AEDBN - Epoch {ep}, loss_train={avg_loss}')
        
        # switch back to evaluation mode
        self.train(False)
    
    def to_clf(self, n_class=5, loss='CrossEntropyLoss', optimizer='Adam', lr=0.01,
               epochs=50, batch_size=50, loss_kwargs={}, optimizer_kwargs=dict()):
        """
        Converts the AEDBN into a classifier by replacing the decoder with a classification layer.

        Parameters:
        ----------
        n_class : int, optional (default=5)
            Number of classes for classification.

        loss : str or class, optional (default='CrossEntropyLoss')
            Loss function to be used for classification. It must be a string exactly equal to 
            the name of a loss function in torch.nn module (e.g., 'MSELoss', 'CrossEntropyLoss', etc.), as you
            are importing the loss function.
        
        optimizer : str, optional (default='Adam')
            The name of the optimizer used to fine-tune the autoencoder. It must be a string exactly equal to the 
            name of an optimizer in torch.optim module (e.g., 'SGD', 'Adam', etc.), as you are importing the 
            optimizer function.

        lr : float, optional (default=0.01)
            Learning rate for the optimizer.

        epochs : int, optional (default=50)
            Number of epochs for training the classification layer.

        batch_size : int, optional (default=50)
            Batch size for training the classification layer.

        loss_kwargs : dict, optional (default={})
            Additional arguments to be passed to the loss function.

        optimizer_kwargs : dict, optional (default={})
            Additional arguments to be passed to the optimizer.

        Returns:
        -------
        None

        See Also:
        --------
        modules.CDBN
        """
        return CDBN(copy.deepcopy(self.encoder_), self.dbn.n_hiddens[-1], n_class=n_class, loss=loss, 
                    optimizer=optimizer, lr=lr, epochs=epochs, batch_size=batch_size,
                    loss_kwargs=loss_kwargs, optimizer_kwargs=optimizer_kwargs, verbose=self.verbose)

class CDBN(nn.Module):
    '''
    A classifier module constructed from the fine-tuned autoencoder.

    Parameters:
    ----------
    encoder : nn.Module
        The fine-tuned encoder.
    
    encode_size : int
        The size of the output encoding of the encoder module.
    
    n_class : int, optional (default=5)
        The number of classes in the classification task.
    
    loss : str or class, optional (default='CrossEntropyLoss')
        Loss function to be used for classification. It must be a string exactly equal to the name of a 
        loss function in torch.nn module (e.g., 'MSELoss', 'CrossEntropyLoss', etc.), as you are importing 
        the loss function.
        
    optimizer : str, optional (default='Adam')
        The name of the optimizer used to fine-tune the autoencoder. It must be a string exactly equal to the 
        name of an optimizer in torch.optim module (e.g., 'SGD', 'Adam', etc.), as you are importing the 
        optimizer function.
    
    lr : float, optional (default=0.01)
        The learning rate used for training the classifier.
    
    epochs : int, optional (default=50)
        The number of epochs used for training the classifier.
    
    batch_size : int, optional (default=50)
        The batch size used for training the classifier.
    
    loss_kwargs : dict, optional (default={})
        Optional arguments for the loss function.
    
    optimizer_kwargs : dict, optional (default={})
        Optional arguments for the optimizer.
    
    verbose : bool, optional (default=True)
        Whether to print training progress.

    '''
    def __init__(self, encoder, encode_size, n_class=5, loss='CrossEntropyLoss', optimizer='Adam', lr=0.01,
                 epochs=50, batch_size=50, loss_kwargs={}, optimizer_kwargs=dict(), verbose=True):
        super(CDBN,self).__init__()

        self.encoder = encoder
        self.encode_size = encode_size
        self.n_class = n_class
        self.loss = loss
        self.optimizer = optimizer
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.loss_kwargs = loss_kwargs
        self.optimizer_kwargs = optimizer_kwargs

        self.output_layer_ = nn.Linear(self.encode_size, n_class)

    def forward(self, X):
        """
        Performs a forward pass through the CDBN.

        Parameters:
        ----------
        X : PyTorch tensor, shape (n_samples, n_features)
            The input data.

        Returns:
        -------
        pred : tensor, shape (n_samples, n_classes)
            The output layer activations.
        """
        X = torch.as_tensor(X, dtype=torch.float)
        return self.output_layer_(self.encoder(X))

    def fine_tune(self, X, y):
        """
        Fine-tunes the pre-trained model using the backpropagation algorithm.

        Parameters:
        ----------
        X : PyTorch tensor of shape (n_samples, n_features)
            The input data.

        y : PyTorch tensor of shape (n_samples,)
            The target values.

        Returns:
        -------
        None
        """
        X = torch.as_tensor(X, dtype=torch.float)

        # Create a DataLoader object to efficiently load the data in batches
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size)

        # Import the loss function and optimizer
        loss_fn = getattr(import_module('torch.nn'), self.loss)(**self.loss_kwargs)
        optimizer = getattr(import_module('torch.optim'), self.optimizer)
        optimizer = optimizer(self.parameters(), lr=self.lr, **self.optimizer_kwargs)

        # Set the model to training mode
        self.train(True)

        # Iterate over the epochs
        for ep in range(self.epochs):
            running_loss = 0.
            
            for i, (batch, y_batch) in enumerate(loader):
                optimizer.zero_grad()
                outputs = self.forward(batch)
                loss = loss_fn(outputs, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(loader)

            if self.verbose:
                print(f'CDBN - Epoch {ep}, loss_train={avg_loss}')
        
        # Set the model to evaluation mode
        self.train(False)
    
    def predict(self, X):
        """
        Predict class labels for input data X.

        Parameters:
        -----------
        X: PyTorch tensor of shape (n_samples, n_features)
            The input data to predict the class labels for.
        
        Returns:
        --------
        y_pred: tensor of shape (n_samples,)
            The predicted class labels for the input data.
        """
        X = torch.as_tensor(X, dtype=torch.float)

        self.eval()
        outputs = self.forward(X)
        softmax = nn.LogSoftmax(dim=1)
        _, y_pred = torch.max(softmax(outputs), dim=1)
        return y_pred
