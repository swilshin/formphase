
from .serialisation import serialiseArray,unserialiseArray

from numpy import (sum,asarray,dot,exp,cumsum,hstack,argsort,
  reshape,mod,zeros,arange,empty,pi,concatenate,c_,diff,newaxis,conj,
  empty_like,zeros_like,ones,multiply,add,array,ravel,where)

from copy import deepcopy

class FourierSeries(object):
  def take(self,cols):
    """Get a FourierSeries that only has the selected columns"""
    other = self.copy()
    other.coef = other.coef[:,cols]
    other.m = other.m[cols]
    return other
  
  def __len__(self):
    return(self.coef.size)
  
  def val( self, phi ):
    """Evaluate fourier series at all the phases phi

       Returns rows corresponding to phi.flatten()
    """
    phi = asarray(phi).flatten()
    phi.shape = (len(phi.flat),1)
    th = phi * self.om
    return dot(exp(1j*th),self.coef)+self.m

  def residuals( self, phi, data ):
    '''
    Given a set of data points data at angle phi
    compute the residual of the fourier series.
    '''
    phi = asarray(phi).flatten()
    phi.shape = (len(phi.flat),1)
    data = asarray(data).flatten()
    data.shape = (len(data.flat),1)
    th = phi * self.om
    # Can do this by
    return(cumsum(hstack([data-self.m,-exp(1j*th)*self.coef[:,0]]),1))

  def integrate( self, z0=0 ):
    """Integrate fourier series, and set the mean values to z0
    """
    self.m[:] = asarray(z0)
    self.coef = -1j * self.coef / self.om.T
    return self

  def getDim( self ):
    """Get dimension of output"""
    return len(self.m)

  def getOrder( self ):
    """Get order of Fourier series"""
    return self.coef.shape[0]/2

  def extend( self, other ):
    """Extend a fourier series with additional output columns from
       another fourier series of the same order

       If fs1 is order k with c1 output colums, and
       fs2 is order k with c2 output columns then the following holds:

       fs3 = fs1.copy().append(fs2)
       assert allclose( fs3.val(x)[:c1], fs1.val(x) )
       assert allclose( fs3.val(x)[c1:], fs2.val(x) )
    """
    assert len(other.om) == len(self.om), "Must have same order"
    self.m = hstack((self.m,other.m))
    self.coef = hstack((self.coef,other.coef))

  def diff( self ):
    """Differentiate the fourier series"""
    self.m[:] = 0
    self.coef = 1j * self.coef * self.om.T
    return self

  def copy( self ):
    """Return copy of the current fourier series"""
    return deepcopy( self )

  def fit( self, order, ph, data ):
    """Fit a fourier series to data at phases phi

       data is a row or two-dimensional array, with data points in columns
    """

    phi = reshape( mod(ph + pi,2*pi) - pi, (1,len(ph.flat)) )
    if phi.shape[1] != data.shape[1]:
      raise IndexError(
        "There are %d phase values for %d data points"
            % (phi.shape[1],data.shape[1]))
    # Sort data by phase
    idx = argsort(phi).flatten()
    dat = c_[data.take(idx,axis=-1),data[:,idx[0]]]
    phi = concatenate( (phi.take(idx),[phi.flat[idx[0]]+2*pi]) )

    # Compute means values and subtract them
    #self.m = mean(dat,1).T
    # mean needs to be computed by trap integration also
    dphi = diff(phi)
    self.m = sum((dat[:,:-1] + dat[:,1:]) * .5 * dphi[newaxis,:], axis = 1) / (max(phi) - min(phi))
    #PDB.set_trace()
    dat = (dat.T - self.m).T
    # Allow 0th order (mean value) models
    order = max(0,order)
    self.order = order
    if order<1:
      order = 0
      self.coef = None
      return
    # Compute frequency vector
    om = zeros( 2*order )
    om[::2] = arange(1,order+1)
    om[1::2] = -om[::2]
    self.om = reshape(om,(1,order*2))
    # Compute measure for integral
    #if any(dphi<=0):
      #raise UserWarning,"Duplicate phase values in data"
    # Apply trapezoidal rule for data points (and add 2 pi factor needed later)
    zd = (dat[:,1:]+dat[:,:-1])/(2.0*2*pi) * dphi
    # Compute phase values for integrals
    th = self.om.T * (phi[1:]-dphi/2)
    # Coefficients are integrals
    self.coef = dot(exp(-1j*th),zd.T)
    return self

  def fromAlien( self, other ):
    self.order = int(other.order)
    self.m = other.m.flatten()
    if other.coef.shape[0] == self.order * 2:
      self.coef = other.coef
    else:
      self.coef = other.coef.T
    self.om = other.om
    self.om.shape = (1,len(self.om.flat))
    return self

  def filter( self, coef ):
    """Filter the signal by multiplication in the frequency domain
       Assuming an order N fourier series of dimension D,
       coef can be of shape:
        N -- multiply all D coefficients of order k by
            coef[k] and conj(coef[k]), according to their symmetry
        2N -- multiply all D coefficients of order k by
            coef[2k] and coef[2k+1]
        1xD -- multiply each coordinate by the corresponding coef
        NxD -- same as N, but per coordinate
        2NxD -- the obvious...
    """
    coef = asarray(coef)
    if coef.shape == (1,self.coef.shape[1]):
      c = coef
    elif coef.shape[0] == self.coef.shape[0]/2:
      if coef.ndim == 1:
        c = empty( (self.coef.shape[0],1), dtype=self.coef.dtype )
        c[::2,0] = coef
        c[1::2,0] = conj(coef)
      elif coef.ndim == 2:
        assert coef.shape[1]==self.coef.shape[1],"Same dimension"
        c = empty_like(self.coef)
        c[::2,:] = coef
        c[1::2,:] = conj(coef)
      else:
        raise ValueError("coef.ndim must be 1 or 2")
    self.coef *= c
    return self

  def bigSum( fts, wgt = None ):
    """[STATIC] Compute a weighted sum of FourierSeries models.
       All models must have the same dimension and order.

       INPUT:
         fts -- sequence of N models
         wgt -- sequence N scalar coefficients, or None for averaging

       OUTPUT:
         a new FourierSeries object
    """
    N = len( fts )
    if wgt is None:
      wgt = ones(N,dtype=complex)/float(N)
    else:
      wgt = asarray(wgt,dtype=complex)
      assert wgt.size==len(fts)

    fm = FourierSeries()
    fm.coef = zeros_like(fts[0].coef,dtype=complex)
    fm.m = zeros_like(fts[0].m,dtype=complex)
    for fs,w in zip(fts,wgt):
      fm.coef += w * fs.coef
      fm.m += w * fs.m
    fm.order = fts[0].order
    fm.om = fts[0].om

    return fm
  bigSum = staticmethod( bigSum )
  
  @staticmethod
  def fourierSeries(coef,om,m,order):
    '''
    Constructs a fourier series out of explicit parameters for the 
    co-efficients, orders, etc. Works with form phase serialisations of 
    arrays as strings
    '''
    phi = FourierSeries()
    phi.order = order
    if type(coef)==str:
      phi.coef=unserialiseArray(coef)
    else:
      phi.coef=coef
    if type(om)==str:
      phi.om=unserialiseArray(om)
    else:
      phi.om=om
    if type(m)==str:
      phi.m=unserialiseArray(m)
    else:
      phi.m=m
    return(phi)
  
  def __mul__(self,x):
    '''
    Multiply two fourier series together, this will change the order of the 
    fourier series
    '''
    nc = multiply.outer(self.coef[:,0],x.coef[:,0])
    oms = array(add.outer(self.om[0],x.om[0]),dtype=int)
    order = oms.max()
    om = zeros( 2*order )
    om[::2] = arange(1,order+1)
    om[1::2] = -om[::2]
    coef = zeros_like(om,dtype='complex128')
    m = 0.
    for c,p in zip(ravel(nc),ravel(oms)):
      if p==0:
        m+=c
      else:
        coef[where(p==om)[0]]+=c
    # Add the constant times the fourier series
    for c,p in zip(self.coef[:,0],self.om[0]):
      coef[where(p==om)[0]]+=x.m*c
    for c,p in zip(x.coef[:,0],x.om[0]):
      coef[where(p==om)[0]]+=self.m*c
    # Add product of constant term
    m += self.m * x.m
    f = FourierSeries()
    f.m = array([m])
    f.order = order
    f.om = array([om])
    f.coef = array([coef]).T
    return(f)
  
  def __repr__(self):
    '''
    Produce a string representation of the fourier series, can be used as 
    a serialisation as it is just the command to instantiate the series.
    '''
    return("FourierSeries.fourierSeries("+
      "coef="+repr(serialiseArray(self.coef))+
      ",om="+repr(serialiseArray(self.om))+
      ",m="+repr(serialiseArray(self.m))+
      ",order="+repr(self.order)+
    ")")