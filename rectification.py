
from .formphaseutil import FourierSeries

from numpy import (array,linspace,unwrap,angle,pi,sum,exp,stack,zeros_like, 
  sign,diff,eye,ones_like,tile,arange)

from copy import deepcopy

class Rectification(object):
  def __init__(self,ordR=10,ordP=None):
    self.ordR = ordR
    if ordP is None:
      self.ordP = self.ordR
    else:
      self.ordP = ordP

  def fit(self,lcyc):
    self.fitR0(lcyc)
    self.fitD0(lcyc)
  
  def fitR0(self,lcyc):
    self.fsR0 = FourierSeries().fit(self.ordR,array(lcyc[:,1]),array([lcyc[:,0]]))
  
  def fitD0(self,lcyc):
    orientation = sign(diff(unwrap(lcyc[...,1])[[0,-1]]))
    thetacyc = linspace(0.,2.*pi,lcyc.shape[0])
    thetares0 = unwrap(lcyc[...,1]-orientation*thetacyc)
    muthetares0 = angle(sum(exp(1j*thetares0)))
    thetares = (thetares0-muthetares0+pi)%(2.*pi)-pi
    self.fsD0 = FourierSeries().fit(self.ordP,array(lcyc[:,1]),array([thetares]))
  
  def transformRadius(self,r):
    R0 = self.fsR0.val(r[...,1]).reshape(r[...,1].shape).real
    p = deepcopy(r)
    p[...,0] = r[...,0]/R0
    return(p)
  
  def transformAngle(self,r):
    D0 = self.fsD0.val(r[...,1]).reshape(r[...,1].shape).real
    p = deepcopy(r)
    p[...,1] = r[...,1]-D0
    return(p)
  
  def __call__(self,r):
    rp = self.transformRadius(r)
    rp = self.transformAngle(rp)
    return(rp)

  def jacobian(self,r):
    R0 = self.fsR0.val(r[...,1]).real.reshape(r[...,1].shape)
    R0p = self.fsR0.copy().diff().val(r[...,1]).real.reshape(r[...,1].shape)
    D0p = self.fsD0.copy().diff().val(r[...,1]).real.reshape(r[...,1].shape)
    J = eye(r.shape[-1]).reshape(
      (r.shape[-1],r.shape[-1])+tuple(ones_like(r.shape[:-1]))
    )
    J = tile(J,r.shape[:-1])
    J = J.transpose(arange(2,len(r.shape)+3)%(len(r.shape)+1))
    J0 = stack([
      stack([1./R0,-(r[...,0]*R0p)/(R0*R0)],-1),
      stack([zeros_like(r[...,1]),1.-D0p],-1)
    ],-2)
    J[...,:2,:2] = J0    
    return(J)

def rectification(lcyc,ordR=10,ordP=None):
  r = Rectification(ordR,ordP)
  r.fit(lcyc)
  return(r)