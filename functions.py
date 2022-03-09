
from numpy import exp,array

def gaussian(r,r0=0.,lda=1.):
  return(
    exp(-((r-r0)*(r-r0))/(2*lda*lda))
  )

def pgaussian(r,r0=0.,lda=1.):
  return(
    -((r-r0)/(lda*lda))*exp(-((r-r0)*(r-r0))/(2*lda*lda))
  )

def gfunc(r,r0s,lda):
  return(array([
    gaussian(r,r0,lda) for r0 in r0s
  ]))

def pgfunc(r,r0s,lda):
  return(array([
    pgaussian(r,r0,lda) for r0 in r0s
  ]))

