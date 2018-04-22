# Boltzmann Machines

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
#training_set[:,0] --ist column of training set ie. user
#nb_users is the max no. of users, nb_movies is the max no. of movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
#(lines are going to be the observations in the network and columns are going to be input nodes of the network)
#basically this will be the list of users where each user is the list of its ratings to all the movies
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        #basically all the index of the  movies rated by the user with id==id_users
        #contains the 2nd column ie. movies id such that ist column ie. userId ==id_users
        id_movies = data[:,1][data[:,0] == id_users] 
        #basically actual ratings given by the user with id==id_users
        #contains the 3rd column ie. rating such that ist column ie. userId ==id_users
        id_ratings = data[:,2][data[:,0] == id_users]
        #initiliaze the  ratings by alla list of 0's(then we update the list by putting the actual rating for the particular movie and keep it 0 if not rated by the user)
        ratings = np.zeros(nb_movies)
        #update the movie rated by the user with the actual ratings
        ratings[id_movies - 1] = id_ratings#since movies_id starts with 1 and indexing in python starts with 0
        #now append this list of ratings to the new data
        new_data.append(list(ratings))
    #finaaly new_data will be the list of users where each user is the list of ratings given by the user to the movies
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors(multi dimensional matrices of single type based on pytorch)
training_set = torch.FloatTensor(training_set) #expects a list of list
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set == 0] = -1 #take all the values of the training set where the value==0 and set the value to -1(-1 means the movie is not rated by the user)
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1  #if the value of training_set>=3 means rating>=3 so it is liked
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creating the architecture of the Neural Network
# we will make it a probablistic graphical model bcos a RBM is actullay a probablistic graphical model
class RBM():
    def __init__(self, nv, nh): #nv is the no. of visible nodes and nh is the no. of hidden nodes
        #parameters we need to initilize the RBM
        #here W is the probablity of visible nodes given the hidden
        self.W = torch.randn(nh, nv) # weights is the torch of size (nh x nv) initialized with some random values according to normal distribution(having mean=0 and variance=1)
        #here a is the probablity of hidden nodes given the visible(here a is 2d tensor ,ist dimension is for batch and second one is for bias)
        #we have to do this bcos func we are going to use will expect in this format
        self.a = torch.randn(1, nh)  #bias for hidden nodes (1 for each hidden node)
        self.b = torch.randn(1, nv)  #bias for visible nodes
        
    
    #sampling the hidden nodes according to the probablities P(h , given  v)(probablity that hidden node==1 given the value of visible node==sigmoid activation
    #we are making a bernoulli RBM bcos we are just predicting a binary outcome
    def sample_h(self, x): #x is the given neuron v in the probablities P(h , given  v) 
        wx = torch.mm(x, self.W.t()) #matrix multiplication of W and x
        activation = wx + self.a.expand_as(wx)  #sigmoid func accepts a linear func==wx+a but we are expanding a as the dimension of wx
        p_h_given_v = torch.sigmoid(activation)
        
        #we are returning a bernoulli sample of given distribution ie probablity
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    
    def sample_v(self, y): # y correeponds to the value of hidden node
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation) #probablity that visible node==1 given the value of hidden node
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    
    #v0 is the our input vector,vk==visible nodes obtained after k sampling(or contrastive divergence iteration)
    #ph0is the vector of probablites that at first iteration value of hidden node=1 given the value of visible node
    #vector of probablites that value of hidden node=1 given the value of visible node after k sampling(or contrastive divergence iteration)(ie.vk)
    def train(self, v0, vk, ph0, phk):
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)
nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv, nh)

# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0  #loss func to use the measure the error,we will be using simple diff in absolute value
    s = 0.          #float counter to normalize the loss, we incremnet it after every epoch 
    for id_user in range(0, nb_users - batch_size, batch_size): #looping over all the users but we are taking the batches of users
        vk = training_set[id_user:id_user+batch_size]  #basically our input batch of users,so it will be a batch of users starting from the given user to the next 100 ones
        v0 = training_set[id_user:id_user+batch_size] #actual values of the ratings given by the batch of the 100 users from he current one
        ph0,_ = rbm.sample_h(v0)   #here '_' is just to make sure that we taking only the ist value return by the func ie. p_h_given_v
        for k in range(10):   #loop for k steps of contrastive divergence
            #in step-1 we want to get first sampled hidden nodes
            _,hk = rbm.sample_h(vk) # we  are taking only the 2nd value return  by the func ie.torch.bernoulli(p_h_given_v)
            #initially vk==v0 but now we are going to update the vk
            #now vk will be sampled visible nodes after the ist step of gibbs sampling
            #basically ist update of our visible nodes ie visible nodes after ist step of sampling
            _,vk = rbm.sample_v(hk)
            #now we dont want to train our model on -ve rating ie movies which are not rated by the given user
            #so we keep the -ve ratings as it is
            vk[v0<0] = v0[v0<0]  #take all the values of vk where the value was -ve in v0 and assign it -1
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk) #training our model to update the values of w,b,a towards the direction of max. likelihood
        
        #updating the train_loss basically absolute diff between our final vk( values of input nodes corresponding to optimal set of weights ) and v0(original) 
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))  #v0>=0 means we are taking only the +ve ratings
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):  #we dont need batch size in test set
    
    #by using the inputs of our training set we activate the hidden neurons of our RBM to predict the ratings of our test set
    v = training_set[id_user:id_user+1]  #input on which we make our predictions ( )
    vt = test_set[id_user:id_user+1]   #target ie. to which we compare our predictions 
    #here we don't need to take 10 steps
    #basically only one iteration of gibbs sampling
    if len(vt[vt>=0]) > 0:   #all the ratings that are existent ie. positive
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))