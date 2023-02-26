import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

class RBM(nn.Module):
    """
    A pytorch implementation of the Bernoulli Restricted Boltzmann Machine (RBM)

    Parameters
    ----------
    n_visible : int
        Number of visible units (input features).
    
    n_hidden : int
        Number of hidden units (learned features).
    
    lr : float, optional (default=1e-5)
        Learning rate for the model.
    
    epochs : int, optional (default=10)
        Number of epochs to train the model for.
    
    batch_size : int, optional (default=30)
        Batch size for training.
    
    k : int, optional (default=3)
        Number of Gibbs sampling steps.
    
    use_gpu : bool, optional (default=True)
        Whether to use GPU if available.
    
    verbose : bool, optional (default=True)
        Whether to print training progress during training.

    Attributes
    ----------
    W : torch tensor shape = (n_visible, n_hidden)
        Weight matrix connecting the visible and hidden units.
    
    vb : torch tensor shape = (n_visible,)
        Bias term for the visible units.
    
    hb : torch tensor shape = (n_hidden,)
        Bias term for the hidden units.

    Methods
    -------
    v_to_h(v)
        Converts the data in visible layer to hidden layer, also does sampling.
    
    h_to_v(h)
        Converts the data in hidden layer to visible layer, also does sampling.
    
    contrastive_divergence(v0)
        Performs contrastive divergence on the input data.
    
    forward(X)
        Passes the input data through the model and returns the hidden layer activations.
    
    train(dataset)
        Trains the model on the input dataset.

    """

    def __init__(self, n_visible, n_hidden, lr=1e-5, epochs=10, batch_size=30, k=3, use_gpu=True, verbose=True):
        super(RBM, self).__init__()

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.k = k
        self.use_gpu = use_gpu
        self.verbose = verbose

        # Set the device to GPU if available
        if torch.cuda.is_available() and use_gpu==True:
            dev = "cuda:0"
        else:
            dev = "cpu"
        self.device_ = torch.device(dev)

        # Initialise weights and biases
        std = 4 * np.sqrt(6. / (self.n_visible + self.n_hidden))
        self.W = torch.normal(mean=0, std=std, size=(self.n_visible, self.n_hidden))
        self.vb = torch.zeros(self.n_visible)
        self.hb = torch.zeros(self.n_hidden)

        self.W = self.W.to(self.device_)
        self.vb = self.vb.to(self.device_)
        self.hb = self.hb.to(self.device_)

    def v_to_h(self, v):
        '''Converts the data in visible layer to hidden layer, also does sampling
        v here is the visible probabilities

        Parameters
        ----------
        v : torch tensor shape = (n_samples , n_features)
            The input visible layer, which contains the probabilities of each visible 
            unit being activated (or the observed data).

        Returns
        -------
        h : torch tensor shape = (n_samples, n_hidden)
            The new hidden layer (probabilities) obtained from the input visible layer.
        
        sample_h : torch tensor shape = (n_samples, n_hidden)
            The Gibbs sampling of the new hidden layer. It contains binary values (0 or 1) based on whether the 
            corresponding hidden unit is activated or not. 
        '''
        h = torch.matmul(v,self.W)   # calculate the activations of hidden units
        h = torch.add(h, self.hb)    # add bias term to the activations
        h = torch.sigmoid(h)
        return h, torch.bernoulli(h) # return both the probabilities and binary samples of hidden layer units
    
    def h_to_v(self, h):
        '''Converts the data in hidden layer to visible layer, also does sampling
        h here is the hiddle probabilities

        Parameters
        ----------
        h : torch tensor shape = (n_samples , n_hidden)
            The input hidden layer, which contains the probabilities of each hidden unit being activated.

        Returns
        -------
        v : torch tensor shape = (n_samples, n_visible)
            The new reconstructed visible layer (probabilities) obtained from the input hidden layer.
        
        sample_v : torch tensor shape = (n_samples, n_visible)
            The Gibbs sampling of the new reconstructed visible layer. It contains binary values (0 or 1) based 
            on whether the corresponding visible unit is activated or not.
        '''
        v = torch.matmul(h,self.W.t())  # calculate the activations of visible units
        v = torch.add(v, self.vb)       # add bias term to the activations
        v = torch.sigmoid(v)
        return v, torch.bernoulli(v)    # return both the probabilities and binary samples of visible layer units
    
    def contrastive_divergence(self, v0):
        '''Perform contrastive divergence algorithm to update the parameters
        
        Parameters
        ----------
        v0 : torch tensor shape = (n_samples, n_visible)
            The input visible layer data used for computing deltas.

        Returns
        -------
        err : float
            The reconstruction error between the input data and the reconstructed data.

        '''

        # initial activations of hidden units and hidden samples using the input data
        h0, hkact = self.v_to_h(v0)

        # perform gibbs sampling k times to get the final samples of hidden and visible units
        for i in range(self.k):
            vk, _ = self.h_to_v(hkact)
            hk, hkact = self.v_to_h(vk)
        
        # compute delta for the parameters using the input and the final samples of hidden and visible units
        dW = torch.mm(v0.t(), h0) - torch.mm(vk.t(), hk) # delta for W
        dvb = torch.sum((v0-vk), 0) # delta for visible unit biases
        dhb = torch.sum((h0-hk), 0) # delta for hidden unit biases

        # update the parameters using the computed deltas
        self.W += self.lr * dW
        self.vb += self.lr * dvb
        self.hb += self.lr * dhb

        # compute the reconstruction error between the input and the final reconstructed visible layer
        err = torch.mean(torch.sum((v0 - vk)**2, 0))
        return err

    def forward(self, X):
        """
        Perform a forward pass through the network, mapping input X to hidden layer activations.

        Parameters
        ----------
        X : torch tensor shape = (n_samples , n_features)
            Input data to be transformed by the RBM.

        Returns
        -------
        h : torch tensor shape = (n_samples , n_hidden)
            The hidden layer activations corresponding to the input X.
        """
        return self.v_to_h(X)

    def train(self, dataset):
        """
        Train the RBM using the specified dataset.

        Parameters
        ----------
        dataset : torch Dataset
            The dataset to use for training the RBM.

        Returns
        -------
        None
        """
        loader = DataLoader(dataset, batch_size=self.batch_size)

        for ep in range(self.epochs):
            running_cost = 0.
            n_batchs = 0
            for i, (batch, _) in enumerate(loader):
                batch = batch.view(len(batch), self.n_visible)
                running_cost += self.contrastive_divergence(batch)
                n_batchs += 1

            if self.verbose:
                print(f'RBM - Epoch: {ep}, averaged cost = {running_cost/n_batchs}')
        return

class GBRBM(RBM):
    '''
    A Gaussian-Bernoulli Restricted Boltzmann Machine (GBRBM).
    Visible layer can assume real values, while hidden layer assumes binary values only.
    '''

    def h_to_v(self,h):
        '''Converts the data in hidden layer to visible layer, also does sampling
        h here is the hiddle probabilities, the visible units follow gaussian distributions

        Parameters
        ----------
        h : torch tensor, shape = (n_samples , n_hidden)
            Hidden layer probabilities.

        Returns
        -------
        v : torch tensor, shape = (n_samples, n_visible)
            New reconstructed layer (probabilities).
        
        sample_v : torch tensor, shape = (n_samples, n_visible)
            Gibbs sampling of new visible layer.
        '''

        v = torch.matmul(h ,self.W.t()) # calculate the activations of visible units
        v = torch.add(v, self.vb)       # add bias term to the activations

        # return both the probabilities and gaussian samples of visible layer units
        return v, v + torch.normal(mean=0, std=1, size=v.shape).to(self.device_) 
    