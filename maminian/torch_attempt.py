import torch
import numpy as np
import aux_files # has well data.

import sklearny as sy # loads data, creates an "X" and "y"

'''
Manuch -- 

This version still does a linear regression, but only on the wells.

'''

class Attempt(torch.nn.Module):

    def __init__(self):
        super(Attempt, self).__init__()

        # random initialization of coefficients to be fit
        self.B = torch.nn.Parameter(torch.rand(sy.dim_in, sy.dim_out)) # coefs in the linear model
        self.b = torch.nn.Parameter(torch.rand(1,1)) # bias/constant shift

    def forward(self, X):
        '''
        What goes here: the input X, and how the output 
        is predicted (whether linear or not). 
        
        NOTHING ELSE SHOULD BE DONE HERE.
        '''
        
        outputs = X @ self.B + self.b
        
        #outputs = 
        
        return outputs


    def fit(self, X, y, loss='MSELoss', lr=0.1, 
        keep_every=None, print_every=None, regpen=1e-4, num_epochs=100,
        batchsize_min=50):
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
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum = 0.1, nesterov=True)

        # if we want to track history.
        keeploss1 = np.zeros(num_epochs)
        keeploss2 = np.zeros(num_epochs)
        keeploss = np.zeros(num_epochs)
        

        blah = []

        for epoch in range(num_epochs):

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # DO NOT TOUCH THESE TWO
            # TODO: possibly do batch (random subsets of wells)
            #X_wells = X[:, aux_files.iuobs]
            #yexact = y[:, aux_files.iuobs]
            ypred = self(X)
            
            # well location indices
            ii = aux_files.iuobs

            # evaluate the actual loss.
            loss1 = criterion(y[:,ii], ypred[:,ii]) # l2 error for the prediction
            
            loss2 = torch.linalg.norm(ypred, 'fro')
            loss = loss1 + regpen*loss2
            #loss = loss1

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters (actual gradient descent step)
            optimizer.step()

            if (epoch) % print_every == 0:
                print(f'It: {epoch}. l2: {loss1}, reg: {loss2}'.format(epoch, loss1, loss2))

            keeploss[epoch] = loss.item()
            keeploss1[epoch] = loss1.item()
            keeploss2[epoch] = loss2.item()
            #wactive[epoch] = sum(abs(self.w) > wthresh) # if using l1 regularization.
            
            if (epoch) % keep_every == 0:
                blah.append(
                {
                'epoch' : epoch,
                'Tpred' : self(X),
                'B' : self.B
#                'loss' : loss.item(),
#                'loss_base' : loss1.item(),
#                'loss_l1' : loss2.item()
                }
                
                )
        
        # evaluate one final call of the model on the entire input X 
        # at the end.
        pred = self(X)
        
        pointwise_err = abs(ypred - y) # TODO: not saved, not outputted.
        
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
    
    # TODO: proper train/test validation.
    
    # Input: xi_ens
    Xtorch = torch.Tensor(sy.X)
    # output: u_ens
    ytorch = torch.Tensor(sy.y)
    
    optimizer = Attempt()
    
    
    ypred_final = optimizer.fit(
        Xtorch[:1], 
        ytorch[:1], 
        regpen=1e-1,
        num_epochs = 1000, 
        print_every=100)
    
    
    
    ypred_np = ypred_final[:1].detach().numpy().flatten()
    yexact_np = ytorch[:1].detach().numpy().flatten()
    
    fig,ax = sy.vis(ypred_np)
    fig2,ax2 = sy.vis(ypred_np - yexact_np)
    
    
    
    #aux_files.overlay_wells(sy.geom, ax)
    fig.show()
    fig2.show()

