'''
Generate random matricies including positive
definite matricies with eigenvalues drawn
from the uniform distribution, .

@author: Simon Wilshin
@contact: swilshin@rvc.ac.uk
@date: Feb 2013
'''

from numpy import random,dot,diag,linalg,eye,array,hstack,ones
from math import floor

def getRandomOrthogonal(n,s):
  '''
  Create a random orthogonal matrix
  '''
  if not n>0:
    raise ValueError("cant create matrix of zero or negative size (n<=0)!")
  if floor(n)!=n:
    raise ValueError("size of generated matrix must be an integer (n not integer)!")
  Q = linalg.qr(random.randn(n,n))[0]
  return(Q)

def getRandomPositiveDef(n,s=1.0,smin=0.0):
  '''
  Creates a random positive definite matrix
  of dimension n, an integer greater than
  zero with eigenvalues less than s. Default
  for max eigenvalue is 1.0.
  Will not produce a matrix with eigenvalue
  less than smin, default value is zero.

  # Create a random 5x5 matrix
  >>> getRandomPositiveDef(5).shape
  (5, 5)

  # Create 10 20x20 matricies and check
  # eigenvalues are all less than s=5
  >>> max([max(linalg.svd(getRandomPositiveDef(20,5.0))[1]) for i in range(10)])<5.0
  True

  '''
  if not n>0:
    raise ValueError("cant create matrix of zero or negative size (n<=0)!")
  if floor(n)!=n:
    raise ValueError("size of generated matrix must be an integer (n not integer)!")
  Q = getRandomOrthogonal(n,s)
  return(dot(dot(Q,diag(smin+(s-smin)*random.rand(n))),Q.T))

def getRandomAffine(n,s=1.0):
  '''
  Create a random affine matrix of size
  n+1 x n+1. Entries are drawn for the
  normal distribution with mean zero
  everywhere except on the diagonal
  where the mean is one, and standard
  deviation is specified by s (default
  is 1.0). Last row of matrix is
  set to make this a valid affine
  transformation.

  # Create an affine matrix for a 5
  # dimensional space (6 x 6 matrix)
  >>> getRandomAffine(5).shape
  (6, 6)

  # Last row of matrix is always
  # an identity transformation in
  # an affine transformation
  >>> getRandomAffine(3)[-1]
  array([ 0.,  0.,  0.,  1.])
  '''
  if not n>=0:
    raise ValueError("cant create matrix of zero or negative size (n<0)!")
  if floor(n)!=n:
    raise ValueError("size of generated matrix must be an integer (n not integer)!")
  M = s*random.randn(n+1,n+1)
  M[-1,:]=0
  M += eye(n+1)
  return(M)

def invAffine(M):
  '''
  Calculate the inverse of an affine matrix.

  # Invert a random affine matrix and check
  # this is the identity to within machine
  # precision.
  >>> M0 = getRandomAffine(5)
  >>> eps = finfo(float).eps
  >>> (dot(invAffine(M0),M0)-eye(6)).sum()<(6*6+1)*eps
  True
  '''
  iA = linalg.inv(M[:-1,:-1])
  iM = eye(M.shape[0])
  iM[:-1,:-1] = iA
  iM[:-1,-1] = -dot(iA,M[:-1,-1])
  return(iM)

def applyAffine(A,x):
  '''
  Apply the affine transformation A to the
  vector x. If x is of size n x m then
  A is of size n+1 x n+1

  # Apply an affine transform A to the
  # vectors in x.
  >>> A = array([[6,3],[0,1]])
  >>> x = array([[3],[5]])
  >>> applyAffine(A,x)
  array([[ 21.],
         [ 33.]])
  '''
  if x.ndim!=2:
    raise ValueError("x must be two dimensional")
  if x.shape[1]+1!=array(A).shape[0] or x.shape[1]+1!=array(A).shape[1]:
    raise ValueError("shape mismatch, A must be a square matrix with dimension equal to the length of x plus one")
  return(dot(A,hstack([x,ones((x.shape[0],1))]).T).T[:,:-1])


if __name__=="__main__":
  import doctest
  doctest.testmod()
