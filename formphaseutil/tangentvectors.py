'''
Tangent vectors are vectors tangent to some curve on some manifold. One 
can be described using a co-ordinate on the manifold and the vector 
in the tangent space at that location on the manifold.

Seriously oversimplifying this is a collection of little arrows at points on a 
surface, so we need the points on the surface, and the coponents of the 
little arrows. The class L{TangentVectors} encapsulates these properties, 
performs basic validation (like checking input arrays aren't transposed), 
stores multiple trajectories and provides various views on the stored 
co-ordinates and derivatives.

@author Simon Wilshin
@email swilshin@rvc.ac.uk
@date Oct 2013
'''

from .serialisation import serialiseArray,unserialiseArray

from numpy import array,hstack
from warnings import warn

class InvalidData(Exception):
  def __init__(self,m):
    Exception.__init__(self,m)

class StateTangentMismatch(InvalidData):
  def __init__(self,m):
    InvalidData.__init__(
      self,
      "Point in the state space, x, and tangent vectors, dx, must be specified by same type and their dimensions must match. "+m
    )

class InvalidDimension(InvalidData):
  def __init__(self,m):
    InvalidData.__init__(
      self,
      "Expect an MxDxN_n list of array, an MxDxN array, or DxN array. Dimensions of data in state space, x, do not match. "+m
    )

# Warnings
class TransposeWarning(Warning):
  def __init__(self,m):
    Warning.__init__(
      self,
      "x has more rows than columns, it is probably transposed. "+m
    )

def inP(w,x,dx):
  '''
  A standard inner product between the differential form w, evaluated at 
  x with the tangent vectors represented by dx. Computes
  
    w(x).dx(x)
  
  assuming the dx and x correspond.
  
  @param w: a differential form
  @type w: Form
  @param x: locations at which to evaluate form
  @type x: array
  @param dx: tangent vectors at x
  @type dx: array
  @return: inner product of w and dx at locations x
  @rtype: array
  '''
  return(sum(w.getForm(x)*dx,0))

class TangentVectors(object):
  '''
  A collection for keeping track of and validating tangent vectors. Expects 
  co-ordinates of the tangent vectors on the manifold and their components, 
  and provides various views of these co-ordinates and components.
  '''
  mutables=['_tx','_tdx']
  def __init__(self,x,dx=None):
    '''
    Build a tangent vector out of a pair of points and tangent vectors to a 
    flow at those points. Expects a three dimensional array with the first 
    index 
    @param x: Co-ordinates on some manifold
    @type x: numpy NxDxM_n array
    @param tangent vectors at the co-ordinates x
    @type: dx numpy NxDxM_n 
    '''
    if type(x)==str:
      self._x = unserialiseArray(x)
      self._dx = unserialiseArray(dx)
    else:
      self._x = x
      self._dx = dx
    self._tx = None
    self._tdx = None
    self.checkInput()

  def _calcFlatForm(self):
    '''
    Training our forms does not require that we know how points in the state 
    space are temporally related but the final fourrier corrections do. This 
    function takes a valid training data set x,dx and returns a new x and dx 
    which are Dx(sum_m N_m) arrays. This removes the unneeded temporal 
    information in the data for the form trainings portion of the formphase
    process.
    The user generally does not need to call this function as it is evaluated 
    automatically when the relevant function calls are made.
    '''
    # Check the input is valid
    if type(self._x)==list or self._x.ndim==3:
      self._tx = hstack(self._x)
      self._tdx = hstack(self._dx)
    else:
      self._tx = self._x
      self._tdx = self._dx

  def getNumPoints(self):
    if self._tx is None:
      self._calcFlatForm()
    return(self._tx.shape[1])

  def checkInput(self):
    '''
    The state space and the tangent space can either be
    as list of numpy arrays or numpy arrays. In either
    case they must meet various conditions to be valid,
    and it can be difficult transpose these inputs
    correctly. This routine checks that the state space
    (and optionally the tangent vector in the state space)
    match the minimal conditions necessary to be valid.
    x: positions in the state space, list of numpy arrays
       or a numpy array
    dx: tangent vectors corresponding to the state space x,
        if set to None only x is checked. Must be a list of
        numpy of numpy arrays or a numpy array.
    returns: nothing
    '''
    # Check that x and dx are compatible
    if self._dx is not None:
      if type(self._x)!=type(self._dx):
        raise InvalidData(
          "x is "+str(self._x.__class__) + 
          "while dx is "+str(self._dx.__class__)+"."
        )
      if type(self._x)==list:
        for i,(ix,idx) in enumerate(zip(self._x,self._dx)):
          if ix.shape!=idx.shape:
            raise StateTangentMismatch(
              "For entry " +str(i) + 
              " x is shaped "+str(ix.shape) + 
              "while dx is "+str(idx.shape)+"."
            )
      else:
        if self._x.shape!=self._dx.shape:
          raise StateTangentMismatch(
            "x is shaped "+str(ix.shape) +
            "while dx is "+str(idx.shape)+"."
          )
    # Check is either MxDxN_m dimensional or a 3D numpy array (where M is
    # number of observations, D is the dimension of the state space, and N_m 
    # is the number of points observed in for observation m. This checks that 
    # either the array is 2D or 3D or that the sub elements of the list are 
    # dimension 2.
    if type(self._x)==list:
      if not all([ix.ndim==2 for ix in self._x]):
        raise InvalidDimension(
          "Dimensions of entries of x are" + 
          str([ix.shape for ix in self._x])+"."
        )
    elif self._x.ndim!=2 and self._x.ndim!=3:
      raise InvalidDimension(
        "Number of dimensions for x is " +
        str(self._x.ndim)+"."
      )
    elif self._x.ndim==2:
      if self._x.shape[1]<self._x.shape[0]:
        warn(TransposeWarning("x has shape" + str(self._x.shape)+"."))

  def flatdot(self,w):
    '''
    Compute the inner product of the tangent vectors with
    the form w.
    w Form A form evaluated at x whose inner product we want to
    calculate
    '''
    if self._tx is None:
      self._calcFlatForm()
    if type(w)==array or type(w)==list:
      TypeError(
        "Expected a differential form to compute inner product, got " + 
        str(type(w))+"."
      )
    return(inP(w,self._tx,self._tdx))

  def getX(self):
    return(self._x)

  def getdX(self):
    return(self._dx)

  def getFlatX(self):
    if self._tx is None:
      self._calcFlatForm()
    return(self._tx)

  def getFlatdX(self):
    if self._tx is None:
      self._calcFlatForm()
    return(self._tdx)

  def getDim(self):
    '''
    Calculates the dimension of the state space for the data in
    x. FormPhase can accept data in arrays of multiple topologies
    so this static method is used to find out the dimension of
    each of these topologies.
    x: sample vectors from the state space whose dimension is desired,
       can be a list or numpy array.
    returns: the dimension of the corresponding state space, an integer
    '''
    if type(self._x)==list:
      return(self._x[0].shape[0])
    if self._x.ndim==3:
      return(self._x.shape[1])
    return(self._x.shape[0])

  def getN(self):
    if self._tx is None:
      self._calcFlatForm()
    return(self._tx.shape[1])
  
  def __repr__(self):
    '''
    Produce a string representation of this tangent vector object, which can be 
    evaluated using eval.
    @return: string serialisation of tangent vector object
    @rtype: str
    '''
    return("TangentVectors("+
      "x="+repr(serialiseArray(self._x))+
      ",dx="+repr(serialiseArray(self._dx))+
    ")")

example_tv = TangentVectors(
  array([[0.1,0.2],[1.0,0.0],[0.4, 0.2]]).T,
  array([[1.0,0.0],[0.0,1.0],[0.7,-0.7]]).T
)
