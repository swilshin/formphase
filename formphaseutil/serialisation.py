'''
Helper functions for working with the serialisation routines. Includes 
functions for converting small arrays into hex strings and vice versa.

@author: Simon Wilshin
@contact: swilshin@rvc.ac.uk
@date: 2014 Feb 18
'''
from __future__ import print_function

from numpy import fromstring,ndarray

from binascii import b2a_hex,a2b_hex
from ast import literal_eval

def serialiseArray(x):
  '''
  Takes an ndimensional array and converts it to a HEX bytestring 
  prefaced by the type and size. Format looks like
  "4,1:float64:4fa2bb10"
  Where the portion before the first colon is the representation of the size 
  tuple, the segment between the colons is the type information and the 
  portion after the last colon is is the bytestring of the arrays contents.
  Treats None as a special case, storing as
  None:None:None
  '''
  if x is None:
    return("None:None:None")
  s = repr(x.shape)
  s+=":"+x.dtype.name+":"
  s+=b2a_hex(x.tostring())
  return(s)

def unserialiseArray(s):
  '''
  Take an array (as represented by a string) serialised by the 
  L{serialiseArray} function and unserialise it.
  '''
  if s=="None:None:None":
    return(None)
  sz,tp,d = s.split(":")
  sz = literal_eval(sz)
  x = fromstring(a2b_hex(d),dtype=tp).reshape(sz)
  return(x)

def checkSerialisation(x):
  '''
  Helper routine for checking if serialisation has worked, where worked in 
  this context means that the dictionary containing member variables has 
  identical contents up to members designated as mutable in the class 
  definition. This is achieved by comparing key by key with numpy arrays 
  treated as special cases and serialisable class members checked 
  recursively.
  '''
  nx = eval(repr(x))
  if x.__dict__.keys()!=nx.__dict__.keys():
    return((x.__dict__.keys(),nx.__dict__.keys()),False)
  for k in x.__dict__:
    if hasattr(x.__class__,"mutables") and not k in x.__class__.mutables:
      if type(x.__dict__[k])==ndarray:
        if not all(x.__dict__[k]==nx.__dict__[k]):
          return(k,False)
      else:
        try:
          if x.__dict__[k]!=nx.__dict__[k]:
            return(k,False)
        except e:
          print(e)
          if not checkSerialisation(x.__dict__[k])[1]:
            return(k,False)
  return(None,True)

class StrToBytes(object):
    def __init__(self, fileobj):
        self.fileobj = fileobj
    def read(self, size):
        return self.fileobj.read(size).encode()
    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()

