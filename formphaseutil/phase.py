from .fourier import FourierSeries

from numpy import (exp,where,logical_and,logical_not,arange,hstack,vstack,
  pi,ones_like,zeros_like,array,isnan,unwrap,polyfit,arctan2,newaxis,diff,
  linspace,floor)
from numpy.lib.stride_tricks import  as_strided
from numpy.linalg import lstsq
from scipy.interpolate import interp1d
from scipy.signal import hilbert

def eventPhase(iyd,n=[1,0],eps=1e-9):
  '''
  Construct an analytic signal like version of the system
  '''
  # Work out the angle of the normal vector
  a = arctan2(n[1],n[0])
  z = (iyd[0,:]+1j*iyd[1,:])*exp(1j*a)
  # Find zero crossings of first co-ordinate
  idx0 = where(logical_and(z.real[:-1]<0,z.real[1:]>0))[0]
  # Produce a two dimensional array which we can be interpolated to 
  # estimate phase by assuming a linear trend between events
  idx2P = idx0-eps
  tme = arange(z.size)
  fv = array(hstack([vstack([idx0,zeros_like(idx0)]),vstack([floor(idx2P),2*pi*ones_like(idx2P)])]))
  fv = array(sorted(fv.T,key=lambda x: x[0])).T
  # Fix edge condition
  if fv[1,0] > pi:
    fv = fv[:,1:]
  # Interpolate phase
  nph = interp1d(*fv,bounds_error=False)(tme)
  # We now have nans on the ends, do a linear interpolation to get average 
  # phase velocity and add on to the ends a linear trend like this
  gdIdx = logical_not(isnan(nph))
  nph[gdIdx]=unwrap(nph[gdIdx])
  m,c = polyfit(tme[gdIdx],nph[gdIdx],1)
  if fv[0,-1]+1!= nph.size:
    nph[int(fv[0,-1]+1):] = nph[int(fv[0,-1])]+m*(arange(nph[int(fv[0,-1]+1):].size)+1)
  if fv[0,0]!=0:
    nph[:int(fv[0,0])] = nph[int(fv[0,0])]-m*(nph[:int(fv[0,0])].size-arange(nph[:int(fv[0,0])].size))
  return(nph)
  
def window(a, w):
  shape = a.shape[:-1] + (a.shape[-1] - w + 1, w)
  st = a.strides + (a.strides[-1],)
  return as_strided(a, shape=shape, strides=st)

def prextend(x,pad):
  y = array(x).copy()
  if y.ndim==1:
    y = y[newaxis,:]
  D = y.shape[0]
  y0 = hstack([y[:,0],y[:,1]-y[:,0]])
  Y = vstack([y[:,pad-1::-1],diff(y[:,pad::-1]),diff(diff(y[:,pad+1::-1]))])
  M = vstack([Y[i:-D:D,-1] for i in xrange(D)])
  dM = vstack([Y[i+D::D,-1] for i in xrange(D)])
  Q = array([lstsq(vstack([M[i,:],dM[i,:]]),y0[i::D])[0] for i in xrange(D)])
  Q = hstack([Q[:,0],Q[:,1]])
  array([sum(Y[i:-D:D].T*Q[i::D],-1) for i in xrange(D)])
  Y0 = array([sum(Y[i:-D:D].T*Q[i::D],-1) for i in xrange(D)])
  y = hstack([Y0,y])
  return(y)

def paddedHilbert(x,pad=2):
  '''
  A variant of the hilbert transform that pads the start and the end with 
  copies of the signal in reverse order to allow decay of the transients.
  '''
  y = array(x).copy()
  p = eventPhase(y)
  idx0 = abs(p-p[0]-pad*2*pi).argmin()
  idx1 = abs(p-p[-1]+pad*2*pi).argmin()
  y = hstack([y[:,:idx0],y,y[:,idx1:]])
  return(hilbert(y)[:,idx0:x.shape[1]+idx0])
