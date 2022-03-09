'''
Kalman filter. Python port of Shai Revzen's MATLAB Kalman filter.

@note: Adapted from code written by Shai Revzen
@author: Simon Wilshin
@contact: swilshin@rvc.ac.uk
@date: 2014 Feb 27
'''
from __future__ import print_function

from numpy import (sqrt,abs,sum,ones,log,dot,array,zeros,eye,pi,unwrap,angle, 
  mean,cov,random,zeros_like,abs,prod,vstack,exp,fliplr,meshgrid,inf)
from numpy.linalg import det,inv,lstsq
from scipy.optimize import fmin

from .serialisation import serialiseArray,unserialiseArray

def gaussian_prob(x,m,C,use_log=False):
  '''
  Evaluate a multivariate gaussian density. Gives the probability of 
  drawing x from a multivariate gaussian density with means m, and 
  covariance C. Will return the log of the probability if use_log is 
  set to True.
  @param x: observed samples
  @type x: array
  @param m: means
  @type m: array
  @param C: covariance matrix
  @type C: array
  @param use_log: returned log probability
  @type use_log: bool
  @returns: probability of observing vectors in x
  @rtype: array
  '''
  d,N = x.shape
  M = m*ones(N)
  denom = ((2*pi)**(d/2.0))*sqrt(abs(det(C)))
  mahal = sum(dot(dot((x-M),inv(C)),(x-M).T),1) # Chris Bregler's trick
  if any(mahal<0):
    raise Exception('mahal < 0 => C is not positive definite')
  if use_log:
    return( -0.5*mahal - log(denom) )
  else:
    return( exp(-0.5*mahal)/denom )

class Kalman(object):
  '''
  Kalman filter and smoother.
  '''
  def __init__(self,A,C,Q,R):
    '''
    Initialsie by setting the state and observation matricies and covariances.
    '''
    # system matrix, as in x[:,t+1] = dot(A,x[:,t])
    if type(A)==str:
      self.A=unserialiseArray(A) 
    else:
      self.A=array(A)
    # observation matrix as in y = dot(C,x)
    if type(C)==str:
      self.C=unserialiseArray(C) 
    else:
      self.C=array(C)
    # the system covariance matrix (positive definite)
    if type(Q)==str:
      self.Q=unserialiseArray(Q) 
    else:
      self.Q=array(Q)
    # the observation covariance (positive definite)
    if type(R)==str:
      self.R=unserialiseArray(R)
    else:
      self.R=array(R)
    self.ss = self.C.shape[1]
    self.os = self.C.shape[0]

  def update(self,xnew,Vnew,y):
    '''
    With previous state, xnew, previous covariance, Vnew, and a new 
    observation, y, calculate the kalman gain and determine the new 
    new state estimate, state covariances and the log likelihood 
    of the observation, along with the expected observation.
    @param xnew: previous state estimate
    @type xnewL array
    @param Vnew: previous state covariance
    @type Vnew: array
    @param y: new observation
    @type y: array
    @returns: tuple with new state covariance, expected observation, new 
      state covariances, new conditional state covariances and the log 
      likelihood of the observation
    @rtype: tuple
    '''
    x0 = dot(self.A,xnew)
    y0 = dot(self.C,x0)
    V0 = dot(self.A,dot(Vnew,self.A.T)) + self.Q
    e = y - y0
    S = dot(dot(self.C,V0),self.C.T) + self.R
    Sinv = inv(S)
    K = dot(dot(V0,self.C.T),Sinv)
    xnew = x0 + dot(K,e)
    Vupd = eye(V0.shape[0]) - dot(K,self.C)
    Vnew =  dot(Vupd, V0)
    VVnew = dot(Vupd,dot(self.A, Vnew))
    LL = gaussian_prob(array([e]), zeros((1,e.shape[0])), S, 1)
    return(xnew,y0,Vnew,VVnew,LL)
  
  def filter(self,y,init_x=None,init_V=None):
    '''
    Given observations, y, an estimate of the initial state, x, and an 
    estimate of the initial covariance, init_V, calculate the corresponding 
    state estimates, covariances, log likelihood of observations and 
    predicted observations.
    @param y: observations
    @type y: array
    @param init_x: initial state estimate
    @type init_x: array
    @param init_V: initial state covariance
    @type init_V: array
    @returns: tuple of state estimates, covariances, conditional covariances, 
    log likelihood of observations and predicted observations
    @rtype: tuple
    '''
    y = array(y)
    T = y.shape[1] # number of obervations to filter
    x = zeros((self.ss,T)) # filtered state means
    V = zeros((self.ss,self.ss,T)) # filtered covariances
    VV = zeros((self.ss,self.ss,T)) # filtered covariances
    yEst = zeros(y.shape) # filtered observations
    loglik = 0 # Log likelihood of observation given filter parameters
    if init_x is not None:
      xnew = init_x
    if init_V is not None:
      Vnew = init_V
    for t in xrange(T):
      xnew,y0,Vnew,VVnew,LL = self.update(xnew,Vnew,y[:,t])
      x[:,t] = xnew;
      yEst[:,t] = y0;
      V[:,:,t] = Vnew;
      VV[:,:,t] = VVnew;
      loglik += LL;
    return(x,V,VV,loglik,yEst)
  
  def smooth(self,y,init_x=None,init_V=None):
    '''
    Given observations y, an estimate of the initial state, x, and an 
    estimate of the initial covariance, init_V, perform Rauch-Tung-Striebel 
    (RTS) kalman smoothing, a two pass smoothing approach in which future 
    values are used to refine the estimates of a kalman filter.
    @param y: observations
    @type y: array
    @param init_x: initial state estimate
    @type init_x: array
    @param init_V: initial state covariance
    @type init_V: array
    @returns: tuple of state estimates, covariances, and conditional 
      covariances
    @rtype: tuple    
    '''
    y = array(y)
    T = y.shape[1] # number of obervations to filter
    
    f_x,f_V,f_VV,loglik,f_yEst = self.filter(y,init_x,init_V)
    
    x = zeros((self.ss,T)) # filtered state means
    V = zeros((self.ss,self.ss,T)) # filtered covariances
    VV = zeros((self.ss,self.ss,T)) # filtered covariances
    
    s_x0 = f_x[:,T-1]
    s_V0 = f_V[:,:,T-1]
    
    x[:,T-1] = s_x0
    V[:,:,T-1] = s_V0
    
    for t0 in xrange(T-2,-1,-1):
      # update values from last iteration
      s_x1 = s_x0
      s_V1 = s_V0      
      
      # obtain filtered values
      f_V0 = f_V[:,:,t0]
      f_V1 = f_V[:,:,t0+1]
      f_VV1 = f_VV[:,:,t0+1]
      
      # Predict x1 using the system function 
      f_x0 = f_x[:,t0]
      x1_ap = dot(self.A, f_x0)
      
      # Predict V1 using the f_V0 value
      V1_ap = dot(self.A, dot(f_V0, self.A.T)) + self.Q
      
      # Compute smoothing Kernel
      J = dot(f_V1, dot(self.A.T, inv( V1_ap )))
      
      # Adjust estimates
      s_x0 = f_x0 + dot(J, ( s_x1 - x1_ap ))
      s_V0 = f_V0 + dot(J, ( s_V1 - V1_ap ))
      s_VV1 = f_VV1 + dot(dot((s_V1 - f_V1),inv(f_V1)),f_VV1)
      
      # Store results
      x[:,t0] = s_x0;
      V[:,:,t0] = s_V0;
      VV[:,:,t0+1] = s_VV1;
    
    return(x,V,VV)
  
  def __repr__(self):
    '''
    Produces a string representation of the filter, can be serialised as 
    this is just the commands one types into the terminal to instantiate 
    the filter.
    @returns: expression which instantiates the kalman filter.
    @rtype: str
    '''
    return(
      "Kalman("+(
        "A="+repr(serialiseArray(self.A))+
        ",C="+repr(serialiseArray(self.C))+
        ",Q="+repr(serialiseArray(self.Q))+
        ",R="+repr(serialiseArray(self.R))
      )+
      ")"
    )

def kal0(x,sv=None,Kdisp=1.0,Nsamp=1000,L=5,Norder=3,pg=1.0,vg=1.0,
           sigma0=1000,N0=200,Prange=8):
  x = x.T
  # Time scale
  if sv is None:
    mux = x-mean(x,0)
    phi = unwrap(angle(mux[:,0]+1j*mux[:,1]))
    sv= 2*pi*x.shape[0]/abs(phi[-1]-phi[0])
  # System matrix
  A =  Kdisp*eye(2*x.shape[1])
  A[:x.shape[1],x.shape[1]:2*x.shape[1]] = eye(x.shape[1])/sv
  
  # Observation matrix
  C = zeros((x.shape[1],2*x.shape[1]))
  C[:x.shape[1],:x.shape[1]] = eye(x.shape[1])
  
  # Observation covariance
  R = cov((x[:-1]-x[1:]).T)/sqrt(2.0)
  
  # System covariance
  idx = random.randint(x.shape[0]-5,size=(Nsamp))
  idx = vstack([idx+i for i in xrange(L)])
  tx = x[idx].reshape(idx.shape[0],-1)
  P = array([[(i-(L-1)/2)**j for i in xrange(L)] for j in xrange(Norder)])
  K = lstsq(P.T,tx)[0]
  s = (cov((tx-dot(P[:-1].T,K[:-1]))[1])-cov((tx-dot(P.T,K))[1]))/cov((tx-dot(P[:-1].T,K[:-1]))[1])
  D = zeros_like(A)
  D[:x.shape[1],:x.shape[1]] = R*pg
  D[x.shape[1]:,x.shape[1]:] = R*vg
  Q = D*s
  return(Kalman(A,C,Q,R))

def kalman(x,pg=1.0,vg=1.0,
           sigma0=1000,N0=200,Prange=8):
  KS = kal0(x)
  
  # Fix initial conditions for log likelihood maximisation
  ix = KS.filter(fliplr(x)[:,-N0:],zeros(2*x.shape[0]),sigma0*eye(2*x.shape[0]))[0][:,-1]
  ix[x.shape[0]:] *=-1
  # Grid search to maximise log likelihood
  X0 = [10**i for i in xrange(-Prange,Prange)]
  X,Y = meshgrid(X0,X0)
  X,Y = X.reshape(X.size), Y.reshape(Y.size)
  np,nL = None,inf
  for ip in zip(X,Y):
    L = kalNegLogLikihood(ip,x,ix)
    if L<nL:
      nL=L
      np=ip
  # Nelder-mead refinement
  p = fmin(kalNegLogLikihood,np,args=(x,ix),maxiter=1e3,maxfun=1e4)
  # Create filter
  KS = kal0(x,pg=abs(p[0]),vg=abs(p[1]))
  
  return(KS)

# Current diagnostic code
#hist(hstack([cdist(shiftedls,iyt[:trainTrun[sysNoise,initNoise,dim]]).min(0) for iyt in sim.yt]),100)
#hist(hstack([cdist(ls,iyt).min(0) for iyt in sim.yt]),100)


def kalNegLogLikihood(p,x,x0,verbose=False,Nmax=200):
  L = -kal0(x[:,:Nmax],pg=abs(p[0]),vg=abs(p[1])).filter(x[:,:Nmax],x0,0.5*eye(x0.size))[-2][0]
  if verbose:
    print(p, L)
  return(L)
  
if __name__=="__main__":
  from numpy import linspace,vstack,sin,cos
  from numpy.random import randn
  from pylab import figure,plot,title,show
  from scipy.optimize import fmin

  # Generate data
  t = linspace(0,100,10000)
  x = vstack([(1+0.3*exp(-0.1*t))*sin(t),(1+0.3*exp(-0.1*t))*cos(t)])
  x += 0.05*randn(*x.shape)  

  # Filter
  KS = kalman(x)

  # Initial conditions
  N0=200
  sigma0=1000
  ix = KS.filter(fliplr(x)[-N0:],[0,0,0,0],sigma0*eye(4))[0][:,-1]
  ix[2:] *=-1
  
  # Smooth and filter
  y = KS.smooth(x,ix,0.5*eye(4))[0][:2]
  yf = KS.filter(x,ix,0.5*eye(4))[0][:2]
  figure()
  plot(*y,alpha=0.5)
  plot(*yf,alpha=0.5)
  plot(*x,alpha=0.5)
  title("smooth")
  show()
