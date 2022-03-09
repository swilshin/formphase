from numpy import  array,ones_like,vstack,mean,std,angle,sum,exp,cov,dot
from numpy.linalg import svd
from scipy.interpolate import UnivariateSpline

def PCA(x,fullOutput=False):
  '''
  Use the SVD to perform PCA on x. If fullOutput is true then will return 
  a tuple with transformed data, eigenvalues and eigenvectors, otherwise 
  just returns transformed system.
  '''
  M = cov(x.T)
  V,D,U = svd(M) # dot(U,dot(diag(D),U.T))==M
  if fullOutput:
    return(dot(U,x.T).T,D,U)
  return(dot(U,x.T).T)
