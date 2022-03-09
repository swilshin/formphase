'''
Convention anything dot is time derivative, transforms contravariantly, anything 
d is exterior derivative, transforms covariantly, p stands for prime, ordinary 
derivative.
'''

from numpy import (
  stack,cos,sin,einsum,concatenate,sqrt,arctan2,zeros_like,array,zeros,arange,
  hstack,exp,unwrap,ones,prod,eye,tile,ones_like,dot,diff,pi
)
from numpy.linalg import inv,lstsq
from scipy.sparse.linalg import cg

#(r*r(r+alpha))/((r+alpha)*(R*R))

################################################################################
# HELPER FUNCTIONS
################################################################################

def polarJacobian(r):
  J = eye(r.shape[-1]).reshape(
    (r.shape[-1],r.shape[-1])+tuple(ones_like(r.shape[:-1]))
  )
  J = tile(J,r.shape[:-1])
  J = J.transpose(arange(2,len(r.shape)+3)%(len(r.shape)+1))
  J0 = stack([
    stack([cos(r[...,1]),-r[...,0]*sin(r[...,1])],-1),
    stack([sin(r[...,1]),r[...,0]*cos(r[...,1])],-1)
  ],-2)
  J[...,:2,:2] = J0
  return(J)

def fromPolar(r):
  return(concatenate([
    stack([r[...,0]*cos(r[...,1]),r[...,0]*sin(r[...,1])],-1),
    r[...,2:]
  ],-1))

def toPolar(x):
  return(concatenate([stack([
    sqrt(x[...,0]*x[...,0]+x[...,1]*x[...,1]),
    arctan2(x[...,1],x[...,0]),
  ],-1), x[...,2:]],-1))

def contravariantFromPolar(r,dr):
  J = polarJacobian(r)
  return(einsum('...ij,...j->...i',J,dr))

def covariantFromPolar(r,dr):
  J = polarJacobian(r)
  return(einsum('...ji,...j->...i',inv(J),dr))

def contravariantToPolar(x,dx):
  J = polarJacobian(toPolar(x))
  return(einsum('...ij,...j->...i',inv(J),dx))
  
def covariantToPolar(x,dx):
  J = polarJacobian(toPolar(x))
  return(einsum('...ji,...j->...i',J,dx))

def drho(r):
  return(concatenate([
    covariantFromPolar(r,array([1.,0.])),
    zeros_like(r[...,2:])
  ],-1))

def dtheta(r):
  return(concatenate([
    covariantFromPolar(r,array([0.,1.])),
    zeros_like(r[...,2:])
  ],-1))

def powRho(rho,Nr):
  return(array([rho**n for n in range(1,Nr)]))

def pPowRho(rho,Nr):
  return(array([n*(rho**(n-1)) for n in range(1,Nr)]))

def makeom(order):
  om = zeros( 2*order )
  om[::2] = arange(1,order+1)
  om[1::2] = -om[::2]
  om = hstack([0.,om])
  return(om)

def fSTheta(theta,order):
  om = makeom(order)
  return(exp(1j*einsum('i,...->i...',om,theta)))
  
def pFSTheta(theta,order):
  om = makeom(order)
  return(1j*einsum('i,i...->i...',om,exp(1j*einsum('i,...->i...',om,theta))))

def VRadialFourier(theta,powrho,fstheta,h):
  if h.size>0:
    h1 = h.reshape((h.shape[0]*h.shape[-1],)+h.shape[1:-1])
    return(concatenate([
      [theta],
      concatenate(einsum('i...,j...->ij...',powrho,fstheta),0),
      concatenate(einsum('i...,j...->ij...',h1,fstheta),0)
    ],0))
  else:
    return(concatenate([
      [theta],
      concatenate(einsum('i...,j...->ij...',powrho,fstheta),0)
    ],0))

def dVRadialFourier(r,powrho,fstheta,h,ppowrho,pfstheta,ph):
    drho = zeros_like(r)
    drho[...,0] = 1.
    dtheta = zeros_like(r)
    dtheta[...,1] = 1.
    
    if h.size>0:
      dz = eye(r.shape[-1]).reshape(
        (r.shape[-1],r.shape[-1])+tuple(ones_like(r.shape[:-1]))
      )
      dz = tile(dz,r.shape[:-1])
      dz = dz.transpose(arange(1,len(r.shape)+2)%(len(r.shape)+1))[2:,...]
      h1 = h.reshape((h.shape[0]*h.shape[-1],)+h.shape[1:-1])
      dV = (concatenate([
        einsum(
          'i...,...k->i...k',
          concatenate(einsum('i...,j...->ij...',ppowrho,fstheta),0),
          drho
        ) + einsum(
          'i...,...k->i...k',
          concatenate(einsum('i...,j...->ij...',powrho,pfstheta),0),
          dtheta
        ),
        concatenate(einsum(
          'im...,m...k->im...k',
          concatenate(einsum('ik...l,jk...->ijlk...',ph,fstheta),0),
          dz
        ),0) + einsum(
          'i...,...k->i...k',
          concatenate(einsum('i...,j...->ij...',h1,pfstheta),0),
          dtheta
        )
      ],0))
    else:
      h1 = h.reshape((h.shape[0]*h.shape[-1],)+h.shape[1:-1])
      dV = (
        einsum(
          'i...,...k->i...k',
          concatenate(einsum('i...,j...->ij...',ppowrho,fstheta),0),
          drho
        ) + einsum(
          'i...,...k->i...k',
          concatenate(einsum('i...,j...->ij...',powrho,pfstheta),0),
          dtheta
        )
      )
    return(concatenate([[dtheta],dV],0))

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

def lstsq_cg(A,y,C=None):
  z = dot(A.T,y)
  M = dot(A.T,A)
  x0 = zeros((M.shape[0],))
  if not C is None:
    x0[-1] = 1.
  return([array([cg(M,z,x0,tol=1e-7)[0]]).T])

################################################################################
# FORMS
################################################################################

class LinearFormPhase(object):
  def __init__(self):
    pass
  
  def __call__(self, x):
    return(self.potential(x))
  
  def form(self, x):
    r = toPolar(x)
    rho = r[...,0]-1.
    theta = r[...,1]
    z = r[...,2:]
    
    # Radial terms
    powrho = powRho(rho,self.Nr)
    ppowrho = pPowRho(rho,self.Nr)
    
    h = powRho(z,2)
    ph = pPowRho(z,2)
    
    # Fourier terms
    fstheta = fSTheta(theta,self.order)
    pfstheta =  pFSTheta(theta,self.order)
    
    # Form components
    dV = dVRadialFourier(r,powrho,fstheta,h,ppowrho,pfstheta,ph)
    return(einsum('i,i...k',hstack([self.orientation,self.K]),dV).real)
  
  def potential(self,x):
    r = toPolar(x)
    rho = r[...,0]-1.
    theta = r[...,1]
    z = r[...,2:]
    
    powrho = powRho(rho,self.Nr)
    h = powRho(z,2)
    
    fstheta = fSTheta(theta,self.order)
    V = VRadialFourier(theta,powrho,fstheta,h)
    return(unwrap(einsum('i,i...',hstack([self.orientation,self.K]),V).real))
  
  def train(self,x,dx,C=1,Nr=5,order=4,orientation=1.,usecg=False):
    self.Nr = Nr
    self.order = order
    self.orientation=orientation
    r = toPolar(x)
    dr = contravariantToPolar(x,dx)
    rho = r[...,0]-1.
    theta = r[...,1]
    z = r[...,2:]
    
    # Radial terms
    powrho = powRho(rho,self.Nr)
    ppowrho = pPowRho(rho,self.Nr)
    
    # Fourier terms
    fstheta = fSTheta(theta,self.order)
    pfstheta =  pFSTheta(theta,self.order)
    
    # Off axis terms
    h = powRho(z,2)
    ph = pPowRho(z,2)
    
    # Form components
    dV = dVRadialFourier(r,powrho,fstheta,h,ppowrho,pfstheta,ph)
    
    # Fit
    lstsq_f = (lambda x,y: lstsq_cg(x,y,C)) if usecg else lstsq
    if C==None:
      M = hstack([
        einsum('i...k,...k->i...',dV[1:],dr).reshape(dV.shape[0]-1,-1).T,
        -ones((prod(x.shape[:-1]),1))
      ])
      y = -einsum('...k,...k->...',self.orientation*dV[0],dr).reshape(-1,1)
      self.K = lstsq_f(
        M,
        y
      )[0][:,0]
      self.C = self.K[-1]
      self.K = self.K[:-1]
    else:
      self.C = C
      M = einsum('i...k,...k->i...',dV[1:],dr).reshape(dV.shape[0]-1,-1).T
      y = C*ones((prod(x.shape[:-1]),1))-einsum('...k,...k->...',self.orientation*dV[0],dr).reshape(-1,1)
      self.K = lstsq_f(
        M,
        y
      )[0][:,0]

  def trainx(self,x,dx=None,C=1,Nr=5,order=4,orientation=1.,usecg=False):
    self.Nr = Nr
    self.order = order
    self.orientation=orientation    
    r = toPolar(x)
    rho = r[...,0]-1.
    theta = r[...,1]
    z = r[...,2:]
    
    # Radial terms
    powrho = powRho(rho,self.Nr)
    
    # Fourier terms
    fstheta = fSTheta(theta,self.order)
    
    # Off axis terms
    h = powRho(z,2)
    
    # Form components
    V = VRadialFourier(theta,powrho,fstheta,h)
    
    dV = diff(V)
    dV[0] = (orientation*dV[0].real+pi)%(2*pi)-pi
    
    M = dV.reshape(-1,dV.shape[-1]).T
    y = ones((M.shape[0],))
    
    K = lstsq(M,y)[0]
    self.K = K/K[0].real

class LinearLocalFormPhase(object):
  def __init__(self):
    pass
  
  def __call__(self, x):
    return(self.potential(x))
  
  def form(self, x):
    r = toPolar(x)
    rho = r[...,0]
    theta = r[...,1]
    z = r[...,2:]

    # Radial terms
    g = gfunc(rho,self.r0s,self.lda)
    pg = pgfunc(rho,self.r0s,self.lda)
    
    # Fourier terms
    fstheta = fSTheta(theta,self.order)
    pfstheta =  pFSTheta(theta,self.order)
    
    # Off axis terms
    h = powRho(z,2)
    ph = pPowRho(z,2)
    
    # Form components
    dV = dVRadialFourier(r,g,fstheta,h,pg,pfstheta,ph)
    return(unwrap(einsum('i,i...->...',hstack([1.,self.K]),dV).real))
  
  def potential(self,x):
    r = toPolar(x)
    rho = r[...,0]
    theta = r[...,1]
    z = r[...,2:]
    
    g = gfunc(rho,self.r0s,self.lda)
    fstheta = fSTheta(theta,self.order)
    h = powRho(z,2)
    h = h.reshape((h.shape[0]*h.shape[-1],)+h.shape[1:-1])
    
    V = VRadialFourier(theta,g,fstheta,h)
    return(unwrap(einsum('i,i...->...',hstack([1.,self.K]),V).real))
  
  def train(self,x,dx,C=1,r0s=[1.0],lda=0.5,order=4,regf=0.,orientation=1.,usecg=False):
    self.r0s = r0s
    self.lda = lda
    self.order = order
    self.orientation=orientation
    r = toPolar(x)
    dr = contravariantToPolar(x,dx)
    rho = r[...,0]
    theta = r[...,1]
    z = r[...,2:]

    # Radial terms
    g = gfunc(rho,self.r0s,self.lda)
    pg = pgfunc(rho,self.r0s,self.lda)
    
    # Fourier terms
    fstheta = fSTheta(theta,self.order)
    pfstheta =  pFSTheta(theta,self.order)
    
    # Off axis terms
    h = powRho(z,2)
    ph = pPowRho(z,2)
    
    # Form components
    dV = dVRadialFourier(r,g,fstheta,h,pg,pfstheta,ph)
    
    # Fit
    lstsq_f = (lambda x,y: lstsq_cg(x,y,C)) if usecg else lstsq
    if C==None:
      if regf>0.:
        self.K = lstsq_f(
          hstack([
            concatenate([
              einsum('i...k,...k->i...',dV[1:],dr).reshape(dV.shape[0]-1,-1).T,
              regf*eye(dV.shape[0]-1),
            ],0),
            -ones((prod(x.shape[:-1]),1))
          ]),
          concatenate([
            -einsum('...k,...k->...',self.orientation*dV[0],dr).reshape(-1,1),
            zeros((dV.shape[0]-1,1))
          ],0)
        )[0][:,0]
      else:
        self.K = lstsq_f(
          hstack([
            einsum('i...k,...k->i...',dV[1:],dr).reshape(dV.shape[0]-1,-1).T,
            -ones((prod(x.shape[:-1]),1))
          ]),
          -einsum('...k,...k->...',self.orientation*dV[0],dr).reshape(-1,1)
        )[0][:,0]
      self.C = self.K[-1]
      self.K = self.K[:-1]
    else:
      self.C = C
      if regf>0.:
        self.K = lstsq_f(
          concatenate([
            einsum('i...k,...k->i...',dV[1:],dr).reshape(dV.shape[0]-1,-1).T,
            regf*eye(dV.shape[0]-1),
          ],0),
          concatenate([
            C*ones((prod(x.shape[:-1]),1))-einsum('...k,...k->...',self.orientation*dV[0],dr).reshape(-1,1),
            zeros((dV.shape[0]-1,1))
          ],0)
        )[0][:,0]
      else:
        self.K = lstsq_f(
          einsum('i...k,...k->i...',dV[1:],dr).reshape(dV.shape[0]-1,-1).T,
          C*ones((prod(x.shape[:-1]),1))-einsum('...k,...k->...',self.orientation*dV[0],dr).reshape(-1,1)
        )[0][:,0]
  
  def trainx(self,x, dx=None, C=1, r0s=[1.0], lda=0.5, order=4, regf=0., orientation=1., usecg=False):
    self.r0s = r0s
    self.lda = lda
    self.order = order
    self.orientation=orientation    
    x = x
    r = toPolar(x)
    rho = r[...,0]-1.
    theta = r[...,1]
    z = r[...,2:]
    
    # Radial terms
    g = gfunc(rho,self.r0s,self.lda)

    
    # Fourier terms
    fstheta = fSTheta(theta,self.order)
    
    # Off axis terms
    h = powRho(z,2)
    
    # Form components
    V = VRadialFourier(theta,g,fstheta,h)
    
    dV = diff(V)
    dV[0] = (orientation*dV[0].real+pi)%(2*pi)-pi
    
    M = dV.reshape(-1,dV.shape[-1]).T
    y = ones((M.shape[0],))
    
    K = lstsq(M,y)[0]
    self.K = K/K[0].real
