'''
A wrapper for the pykalman package for automatically training and setting 
initial conditions for smoothing.

@author: Simon Wilshin
@contact: swilshin@rvc.ac.uk
@date: 2014 Feb 10
'''
from __future__ import print_function

try:
  from pykalman import KalmanFilter
except ImportError:
  print("pykalman missing -- feature disabled")
  def KalmanFilter(*arg,**kw):
    raise RuntimeError("pykalman support is missing")

from numpy.linalg import lstsq
from numpy import (pi,random,cov,zeros_like,angle,unwrap,eye,mean,zeros,
  vstack,array,dot)
from scipy.signal import hilbert


def kalmanWrapper(x=None,sv=None,Kdisp=1.0,**args):
    # If not specified generate sensible transition and observation 
    # matricies.
    if sv is None:
      sv= 2*pi*x.shape[0]/unwrap(angle(hilbert(x[:,0]-mean(x[:,0]))))[-1]
    if not args.has_key("transition_matrices"):
      T =  Kdisp*eye(2*x.shape[1])
      T[:x.shape[1],x.shape[1]:2*x.shape[1]] = eye(x.shape[1])/sv
      args["transition_matrices"] = T
    if not args.has_key("observation_matrices"):
      P = zeros((x.shape[1],2*x.shape[1]))
      P[:x.shape[1],:x.shape[1]] = eye(x.shape[1])
      args["observation_matrices"] = P
    if not args.has_key("observation_covariance"):
      args["observation_covariance"] = cov((x[:-1]-x[1:]).T)
    if not args.has_key("transition_covariance"):
      Nsamp = 1000
      L = 2*2+1
      Norder = 3
      idx = random.randint(x.shape[0]-5,size=(Nsamp))
      idx = vstack([idx+i for i in xrange(L)])
      tx = x[idx].reshape(idx.shape[0],-1)
      Q = array([[(i-(L-1)/2)**j for i in xrange(L)] for j in xrange(Norder)])
      K = lstsq(Q.T,tx)[0]
      s = (cov((tx-dot(Q[:-1].T,K[:-1]))[1])-cov((tx-dot(Q.T,K))[1]))/cov((tx-dot(Q[:-1].T,K[:-1]))[1])
      C= zeros_like(T)
      C[:x.shape[1],:x.shape[1]] = args["observation_covariance"]
      C[x.shape[1]:,x.shape[1]:] = args["observation_covariance"]
      args["transition_covariance"] = C*s
    return(KalmanFilter(**args))

