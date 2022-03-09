'''
Routines for taking data and reflecting it
in an ellipse.

@author: Simon Wilshin
@contact: swilshin@rvc.ac.uk
@date: Feb 2013
'''
from __future__ import print_function

from numpy import (sum,dot,array,zeros,eye,arange,linalg,diag,sqrt,
  fromstring,einsum)

from .matrix import getRandomPositiveDef
from .nsphere import generateNSphereAngles,nSphere
from .serialisation import serialiseArray,unserialiseArray

class EllipseReflector(object):
  '''
  A generalised inversion and projection map in which
  the inversion is performed in an ellipse.
  '''
  def __init__(self,A,D2,beta=1.0,m=1.0):
    '''
    This map uses a positive definite matrix to map from a space of size D 
    to one of size D2.

    If D2 is greater than D it projects up by cycling up through the 
    co-ordinates multiple times, otherwise it completes a partial cycle.
    >>> A = array([[4,2],[2,3]])
    >>> y = array([[5,1],[4,3],[3,2],[2,3],[1,5]])
    >>> R = EllipseReflector(A,2)
    >>> R(y)[0]
    array([ 5.04098361,  1.00819672,  5.04098361,  1.00819672])
    '''
    if type(A)==str:
      self.A = unserialiseArray(A)
    else:
      self.A = A
    D = self.A.shape[0]
    if D2<=D:
      self.P = eye(D)[:D2,:]
    else:
      self.P = zeros((D2,D2))[:,:D]
      self.P[arange(D2),arange(D2)%D]=1
    self.m = m
    self.beta = beta
  
  def _r0(self,y):
    return(sum(y*(dot(self.A,y.T).T),1))
  
  #r+a/((1-a)*r*r-2*(1-a)*r+1)-a
  
  def _D(self,r):
    return(r*r+self.beta*r+1)
  
  def _f(self,r):
    D = self._D(r)
    return(self.m*(self.beta+2)*r/D)
  
  def _df(self,r):
    D = self._D(r)
    return(self.m*(self.beta+2)*(1-r*r)/(D*D))
  
  def __call__(self,y):
    r = self._r0(y)
    f = self._f(r)
    yp = dot(self.P,y.T)
    return((f*yp).T)
  
  def derv(self,y,eps=1e-9):
    '''
    Calculate jacobian of transformation. Only for symmetric A
    '''
    if sum((self.A.T-self.A)**2)>eps:
      raise Exception("Derivative of ellipse reflector not implemented for A not symmetric")
    r = self._r0(y)
    f = self._f(r)
    df = self._df(r)
    return((
      einsum('ij,k->ijk',self.P,f) + 
      df*(einsum('ik,jk->ijk',dot(self.P,y.T),2.0*dot(self.A,y.T)))
    ).T)
  
  def getEllipse(self,N=100):
    U,D,V = linalg.svd(self.A)
    B = dot(U,dot(diag(sqrt(D)),V))
    theta = generateNSphereAngles(self.A.shape[0],N)
    return(dot(linalg.inv(B),nSphere(theta).T))
  
  def __repr__(self):
    '''
    Produces a string representation of the reflector, can be serialised as 
    this is just the commands one types into the terminal to instantiate 
    the reflector.
    @returns: expression which instantiates the ellipse reflector.
    @rtype: str
    '''
    return(
      "EllipseReflector("+(
        "A="+repr(serialiseArray(self.A))+
        ",D2="+repr(self.P.shape[0])+
        ",beta="+repr(self.beta)+
        ",m="+repr(self.m)
      )+
      ")"
    )

class EllipseReflectorFactory(object):
  '''
  Factory for random ellipse reflectors
  '''
  def __init__(self,invMax=1.05,invMin=0.95,beta=-1.0,m=0.1,scaleFactor=0,yielded=0):
    '''
    Fixed parameters and those for randomly generated parts of the 
    ellipse reflectors are set at construction.
    '''
    self.invMax = invMax
    self.invMin = invMin
    self.beta = beta
    self.m = m
    self.scaleFactor = scaleFactor
    self.yielded = yielded
    
  def __call__(self,D,D2):
    A = getRandomPositiveDef(D,self.invMax**(1.0+self.yielded*self.scaleFactor),self.invMin**(1.0+self.yielded*self.scaleFactor))
    self.yielded += 1
    return(EllipseReflector(A,D2,self.beta*(1.0+self.yielded*self.scaleFactor*self.scaleFactor),self.m))
  
  def __repr__(self):
    return(
      "EllipseReflectorFactory("+(
        "invMax="+repr(self.invMax)+
        ",invMin="+repr(self.invMin)+
        ",beta="+repr(self.beta)+
        ",m="+repr(self.m)+
        ",scaleFactor="+repr(self.scaleFactor)+  
        ",yielded="+repr(self.yielded)        
      )+
      ")"
    )


if __name__=="__main__":
  import doctest
  doctest.testmod()

  import pylab as pl
  from os.path import join as osjoin
  from simpletestdata import simpleTestData
  '''
  Illustrates how the ellipse mirror is supposed
  to work by distorting an oscillation.
  '''
  # Generate some very simple but shifted test
  # data.
  y = simpleTestData(T=[0.1,0.0])
  # Manually specify the shape to reflect in
  A = array([[42.2,0.3],[0.3,4.7]])
  R = EllipseReflector(2,2,A=A)
  yp = R(y)
  # Get ellipse
  ex,ey = R.getEllipse()
  pl.plot(ex,ey,c='k')
  pl.scatter(*y.T,c='r',alpha=0.3)
  pl.scatter(*yp.T,c='b',alpha=0.3)
  pl.xlabel("horizontal position (arb)")
  pl.ylabel("vertical position (arb)")
  pl.savefig(osjoin("..","..","graphics","ellipseReflectExample.png"))
  
  # Check jacobian of ellipse reflector
  delta = 1e-9
  Delta = delta*eye(2)
  R = EllipseReflector(2,2,q=2.0)
  y0 = array([[0.01,0.01]])
  print(R.derv(y0) - array([(R(y0+d)-R(y0))/delta for d in Delta]))
  
  y0 = array([[0.11,0.22,0.33,0.44]])
  er = EllipseReflector(D2=4,D=4,q=2)
  delta = 1e-9
  Delta = delta*eye(4)  
  print(array([(er(y0+d)-er(y0))/delta for d in Delta])[:,0,:]-er.derv(y0))
  er = EllipseReflector(D2=4,D=3,q=1.3)
  print(array([(er(y0[:,:3]+d[:3])-er(y0[:,:3]))/delta for d in Delta])[:3,0,:]-er.derv(y0[:,:3]))
  
  beta0 = -10.0
  beta1 = -0.1
  for beta in linspace(beta0,beta1,10):
    erf = EllipseReflectorFactory(beta=-beta)
    er = erf(1,1)
    plot(er(array([linspace(0,10,1000)]).T),c=cm.jet((beta-beta0)/(beta1-beta0)))
