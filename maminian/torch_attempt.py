import torch
import torch.nn.functional as F # has ReLU in it.

import numpy as np
import aux_files # has well data.

import sklearny as sy # loads data, creates an "X" and "y"




class Attempt(torch.nn.Module):

    def __init__(self):
        super(Attempt, self).__init__()
        
        # linear shallow thing (low rank rep)
        #
        # (X B_enc + b1) B_dec + b2
        # latent dimension d. 
        m1 = sy.dim_in
        d = 3 # dunno (latent space dim)
        m2 = sy.dim_out        
        
        # random initialization of coefficients to be fit

        #self.activation = torch.tanh
        self.activation = torch.relu
        #self.activation = self.identity
        
        self.B_enc = torch.nn.Parameter(torch.randn(sy.dim_in, d)) # coefs in the linear model
        self.B_dec = torch.nn.Parameter(torch.randn(d, sy.dim_out)) # coefs in the linear model
        
        self.b1 = torch.nn.Parameter(torch.randn(1,d)) # bias/constant shift
        self.b2 = torch.nn.Parameter(torch.randn(1, sy.dim_out))
        #self.b3 = torch.nn.Parameter(torch.rand(1, 1))

    def identity(self, thing):
        return thing
    
    def forward(self, X):
        '''
        What goes here: the input X, and how the output 
        is predicted (whether linear or not). 
        
        NOTHING ELSE SHOULD BE DONE HERE.
        '''
        
        #outputs = X @ self.B + self.b
        
        #outputs = (X @ self.B_enc + self.b1) @ self.B_dec + self.b2
        
        outputs = self.activation( X @ self.B_enc + self.b1 )
        outputs = self.activation( outputs @ self.B_dec + self.b2 )
        #outputs = outputs @ self.B_lin + self.b3
        #outputs = outputs + self.b3
        
        return outputs


    def fit(self, X, y, loss='MSELoss', lr=0.1, 
        keep_every=None, print_every=None, regpen=1e-4, num_epochs=100):
        '''
        What goes on here: the actual training. 
        Give this THE ENTIRE DATA SET.
        
        A thousand knobs to experiment with here.
        '''
        
        if keep_every is None:
            keep_every = np.inf
        if print_every is None:
            print_every = np.inf
        
        # initialization of some type of base minimization
        # default is MSELoss (mean square error)
        criterion = getattr(torch.nn, loss)()

        # minimization method; here "SGD" isn't strictly 
        # stochastic gradient descent, but this represents a class of 
        # minimization techniques based on gradient descent.
        # check torch documentation online for details.
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        # if we want to track history.
        keeploss1 = np.zeros(num_epochs)
        keeploss2 = np.zeros(num_epochs)
        keeploss = np.zeros(num_epochs)
        

        blah = []

        for epoch in range(num_epochs):

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # TODO: possibly do batch (random subsets of wells)
            #X_wells = X[:, aux_files.iuobs]
            #yexact = y[:, aux_files.iuobs]
            ypred = self(X)
            
            # well location indices
            ii = aux_files.iuobs

            # evaluate the actual loss.
            if True:
                loss1 = criterion(y[:,ii], ypred[:,ii]) # l2 error for the prediction
            else:
                loss1 = criterion(y, ypred) # l2 error for the prediction
                
            
            loss2 = torch.linalg.norm(self.B_enc, 'fro') + torch.linalg.norm(self.B_dec, 'fro') # proper choices of loss ... ???
            #loss2 = 0
            loss = loss1 + regpen*loss2
            #loss = loss1

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters (actual gradient descent step)
            optimizer.step()

            if (epoch) % print_every == 0:
                print(f'It: {epoch:.2e}. loss: {loss:.2e}'.format(epoch, loss))

            keeploss[epoch] = loss.item()
            keeploss1[epoch] = loss1.item()
#            keeploss2[epoch] = loss2.item()
            #wactive[epoch] = sum(abs(self.w) > wthresh) # if using l1 regularization.

            # mechanisms to save the history throughout optimization            
            if (not np.isinf(keep_every)) and ((epoch) % keep_every == 0):

                blah.append(
                {
                'epoch' : epoch,
                'predicted' : self(X),
                #'B' : self.B
#                'loss' : loss.item(),
#                'loss_base' : loss1.item(),
#                'loss_l1' : loss2.item()
                }
                
                )
        
        # evaluate one final call of the model on the entire input X 
        # at the end.
        pred = self(X)
        
        #pointwise_err = abs(ypred - y) # TODO: not saved, not outputted.
        
        self.history = blah
        self.keeploss = keeploss   # overall error
        self.keeploss1 = keeploss1 # l2 error
        self.keeploss2 = keeploss2 # regularization term
        #self.wactive = wactive

        return pred
    
    def well_error(self, y1, y2):
        '''
        Returns pointwise error between two torch arrays y1 and y2 
        evaluated at the wells
        
        Inputs:
            y1, y2: torch Tensors; assumed shape (1475,)
        Outputs:
            err : numpy array, dimension len(sy.iuobs) (323,)
        '''
        return (y1.detach().numpy() - y2.detach().numpy() )[:, aux_files.iuobs]

#######


if __name__=="__main__":
    from matplotlib import pyplot as plt
    
    # Input: xi_ens
    Xtorch = torch.Tensor(sy.X)

    # output: u_ens
    ytorch = torch.Tensor(sy.y)

    # TODO: normalization of data within the model.
     
    optimizer = Attempt()
    
    # TODO: proper randomized train/test split
    # TODO: possibly multiple rounds of training at this level
    _ = optimizer.fit(
        Xtorch[:1000], 
        ytorch[:1000], 
        regpen=1e0,
        num_epochs = 10000, 
        print_every=100,
        lr=1e-1)
    
    # evaluate outside training data
    another = 1001
    ypred_test = optimizer(Xtorch[another])
    
    
    
    ypred_np = ypred_test.detach().numpy().flatten()
    yexact_np = ytorch[another].detach().numpy().flatten()
    
    fig,ax = plt.subplots(1,3, figsize=(9,3), sharex=True, sharey=True, constrained_layout=True)
    #fig,ax = sy.vis(ypred_np)
    #fig2,ax2 = sy.vis(ypred_np - yexact_np)
    sy.vis(ypred_np, figax=(fig,ax[0]))
    sy.vis(yexact_np, figax=(fig,ax[1]))
    
    # todo - unwrap visualization into more customizable code.
    sy.vis( (ypred_np - yexact_np)/np.linalg.norm(yexact_np, np.inf), figax=(fig,ax[2]))
    ax[0].set_title(r'$u_\mathrm{pred}$', loc='left', fontsize=18)
    ax[1].set_title(r'$u_\mathrm{exact}$', loc='left', fontsize=18)
    ax[2].set_title(r'$|u_p - u_e|/||u_e||_\infty$', loc='left', fontsize=18)
    
    #aux_files.overlay_wells(sy.geom, ax)
    
    if False:
        import datetime
        tstamp = datetime.datetime.now().strftime("%d%b_%H%M")
        fname_base = "torch_results_" + tstamp
        
        fig.savefig(fname_base + ".png", bbox_inches='tight')
        fig.savefig(fname_base + ".pdf", bbox_inches='tight')
    
    fig.show()
