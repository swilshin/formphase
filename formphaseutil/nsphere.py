'''
Generate random matricies including positive 
definite matricies with eigenvalues drawn 
from the uniform distribution, .

@author: Simon Wilshin
@contact: swilshin@rvc.ac.uk
@date: Feb 2013
'''

from numpy import linspace,array,pi,meshgrid,cos,sin,ones,arange

from math import factorial

def nSphereVolume(n):
  '''
  Calculate the volume of a unit n-sphere. 

  TYPICAL USAGE
  =============
  
    >>> nSphereVolume(0)
    1
    >>> nSphereVolume(1)
    2
    >>> nSphereVolume(2)/pi
    1.0
    >>> nSphereVolume(3)/(pi**2)
    0.5
  
  REFERENCES
  ==========
  
    Equation 5.19.4, NIST Digital Library of Mathematical Functions. http://dlmf.nist.gov/, Release 1.0.6 of 2013-05-06.
  
  @param n: dimension of the n-sphere
  @type n: int
  @return: volume of the unit n-sphere
  @rtype: 
  '''
  if n%2 == 0:
    d = n/2
    return(pi**d/factorial(d))
  else:
    d = (n-1)/2
    return((2*((4*pi)**d)*factorial(d))/(factorial(2*d+1)))


def generateNSphereAngles(N,D):
  '''
  Uniformly samples the spherical polar 
  co-ordinates of the N sphere, not that 
  uniform sample of the co-ordinates 
  does not uniformly sample the spheres
  surface by area.
  Returns an DxN-1 array of the co-ordinates,
  if N=1 then returns points corresponding 
  to -1,1.
  '''
  if N==1:
    '''
    Special case, a sphere in 1D a sphere is 
    just the points -1,1, so return something 
    whose cosine is these.
    '''
    return(array([pi,0]))
  phi = linspace(0,pi,D)
  if N==2:
    '''
    Meshgrid only works in 2+ dimensions so 
    treat the other special case here.
    '''
    return(array([2*phi]).T)
  # All co-ordinates sampled between 0 and pi 
  # except the last one which is sampled 
  # between 0 and 2*pi.
  psi = linspace(0,2*pi,D)
  allPhi = [phi]*(N-2)
  allPhi.append(psi)
  grid = array(meshgrid(*allPhi))
  return(grid.reshape(N-1,grid.size/(N-1)).T)

def nSphere(x):
  '''
  Given a set of angles (format DxN-1, N is N
  of N-Sphere, D is number of points on 
  sphere total, this generates the cartesian 
  co-ordiantes of these points on the sphere.
  If we are dealing with the special 1D case 
  then we need to wrap it so it has the 
  correct layout.
  '''
  if x.ndim==1:
    return(array([cos(x)]).T)
  return(array([nSphereCord(x,i) for i in arange(x.shape[1]+1)]).T)

def nSphereCord(x,m):
  '''
  Generates the mth co-ordinate from the 
  spherical polar angles in x.
  Should be of the form
  sin(x[:,0])*...*sin(x[:,m-1])*cos(x[:,m])
  Unless m=0 then it is is
  cos(x[:,0])
  or m=x.shape[1]+1 when we have just
  sin(x[:,0])*...*sin(x[:,m-1])
  '''
  res = ones(x.shape[0])
  for n in range(x.shape[1]):
    if n<m:
      res*=sin(x[:,n])
    if n==m:
      res*=cos(x[:,n])
  return(res)
