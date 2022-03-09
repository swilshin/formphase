'''
Generates torii.

@author: Simon Wilshin
@contact: swilshin@rvc.ac.uk
@date: Apr 2013
'''

from numpy import (linspace,meshgrid,array,cos,sin,pi,sqrt,vstack,eye,
  ones_like,zeros_like,sum)
from numpy.random import rand,randn


def createTorus(r0=1.0, r1=0.06,DD=10):
  theta0 = linspace(0,2*pi,DD)
  theta1 = linspace(0,2*pi,DD)

  theta0,theta1 = meshgrid(theta0,theta1)
  X = (r0+r1*cos(theta1))*cos(theta0)
  Y = (r0+r1*cos(theta1))*sin(theta0)
  Z = r1*sin(theta1)
  return(array([X,Y,Z]).reshape(3,DD*DD).T)

def sampleSCrossSD(r0=1.0,r1=0.1,D=2,N=100):
  '''
  Make this sample a sensible number of points at each theta (so you need 
  2*pi/dtheta points in theta where dtheta is the scale of the angular
  structure and 2*D*log(D) points at each angle).
  '''
  theta = 2*pi*rand(N)
  x = randn(D,N)
  x /= sqrt(sum(x**2,0))/r1
  x = vstack([[zeros_like(x[0])],x])
  x[1] += r0*ones_like(x[0])
  R = eye(D+1).repeat(N).reshape(D+1,D+1,N)
  R[0,0] = cos(theta)
  R[1,1] = cos(theta)
  R[1,0] = sin(theta)
  R[0,1] = -sin(theta)
  x = sum(R*x,1)
  return(x.T)
