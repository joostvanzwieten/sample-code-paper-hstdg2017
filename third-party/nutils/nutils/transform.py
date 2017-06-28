# -*- coding: utf8 -*-
#
# Module ELEMENT
#
# Part of Nutils: open source numerical utilities for Python. Jointly developed
# by HvZ Computational Engineering, TU/e Multiscale Engineering Fluid Dynamics,
# and others. More info at http://nutils.org <info@nutils.org>. (c) 2014

"""
The transform module.
"""

from . import cache, numeric, core, _
import numpy


class TransformChain( tuple ):

  __slots__ = ()

  @property
  def todims( self ):
    return self[0].todims

  @property
  def fromdims( self ):
    return self[-1].fromdims

  @property
  def isflipped( self ):
    return isflipped( self )

  def __lshift__( self, other ):
    # self << other
    assert isinstance( other, TransformChain )
    if not self:
      return other
    if not other:
      return self
    assert self.fromdims == other.todims
    return TransformChain( self + other )

  def __rshift__( self, other ):
    # self >> other
    assert isinstance( other, TransformChain )
    return other << self

  @property
  def flipped( self ):
    assert self.todims == self.fromdims+1
    return self.__class__( trans.flipped if trans.todims == trans.fromdims+1 else trans for trans in self )

  @property
  def det( self ):
    det = 1
    for trans in self:
      det *= trans.det
    return det

  @property
  def ext( self ):
    ext = numeric.ext( self.linear )
    return ext if not self.isflipped else -ext

  @property
  def offset( self ):
    return offset( self )

  @property
  def linear( self ):
    return linear( self )

  def apply( self, points ):
    return apply( self, points )

  def solve( self, points ):
    return numeric.solve_exact( self.linear, (points - self.offset).T ).T

  def __str__( self ):
    return ' << '.join( str(trans) for trans in self ) if self else '='

  def __repr__( self ):
    return '{}( {} )'.format( self.__class__.__name__, self )

  @property
  def flat( self ):
    return self if len(self) == 1 \
      else affine( self.linear, self.offset, isflipped=self.isflipped )

  @property
  def canonical( self ):
    # Keep at lowest ndims possible. The reason is that in this form we can do
    # lookups of embedded elements.
    items = list( self )
    for i in range(len(items)-1)[::-1]:
      trans1, trans2 = items[i:i+2]
      if mayswap( trans1, trans2 ):
        trans12 = TransformChain(( trans1, trans2 )).flat
        try:
          newlinear, newoffset = numeric.solve_exact( trans2.linear, trans12.linear, trans12.offset - trans2.offset )
        except numpy.linalg.LinAlgError:
          pass
        else:
          trans21 = TransformChain( (trans2,) + affine( newlinear, newoffset ) )
          assert trans21.flat == trans12
          items[i:i+2] = trans21
    return CanonicalTransformChain( items )

  def promote( self, ndims ):
    head = self.canonical
    tail = ()
    while head.fromdims < ndims:
      head, tmp = head.promote_helper()
      tail = tmp + tail
    return head, CanonicalTransformChain(tail)

  def lookup( self, transforms ):
    if not transforms:
      return
    for trans in transforms:
      ndims = trans.fromdims
      break
    head, tail = self.canonical.promote( ndims )
    while head:
      if head in transforms:
        return CanonicalTransformChain(head), TransformChain(tail)
      tail = head[-1:] + tail
      head = head[:-1]

  def lookup_item( self, transforms ):
    head_tail = self.lookup( transforms )
    if not head_tail:
      raise KeyError( self )
    head, tail = head_tail
    item = transforms[head] if isinstance( transforms, dict ) \
      else transforms.index( head )
    return item, tail

class CanonicalTransformChain( TransformChain ):

  __slots__ = ()

  @property
  def canonical( self ):
    return self

  def promote_helper( self ):
    index = core.index( trans.fromdims == self.fromdims for trans in self )
    head = self[:index]
    uptrans = self[index]
    if index == len(self)-1 or not mayswap( self[index+1], uptrans ):
      tail = self[index:]
    else:
      for i in range( index+1, len(self) ):
        scale = self[i]
        if not mayswap( scale, uptrans ):
          break
        head += Scale( scale.linear, uptrans.apply(scale.offset) - scale.linear * uptrans.offset ),
      else:
        i = len(self)+1
      assert equivalent( head[index:]+(uptrans,), self[index:i] )
      tail = (uptrans,) + self[i:]
    return CanonicalTransformChain(head), CanonicalTransformChain(tail)

mayswap = lambda trans1, trans2: isinstance( trans1, Scale ) and trans1.linear == .5 and trans2.todims == trans2.fromdims + 1 and trans2.fromdims > 0


## TRANSFORM ITEMS

class TransformItem( cache.Immutable ):

  def __init__( self, todims, fromdims ):
    self.todims = todims
    self.fromdims = fromdims

  __lt__ = lambda self, other: id(self) <  id(other)
  __gt__ = lambda self, other: id(self) >  id(other)
  __le__ = lambda self, other: id(self) <= id(other)
  __ge__ = lambda self, other: id(self) >= id(other)

  def __repr__( self ):
    return '{}( {} )'.format( self.__class__.__name__, self )

class Shift( TransformItem ):

  def __init__( self, offset ):
    self.linear = self.det = numpy.array(1.)
    self.offset = offset
    self.isflipped = False
    assert offset.ndim == 1
    TransformItem.__init__( self, offset.shape[0], offset.shape[0] )

  def apply( self, points ):
    return points + self.offset

  def __str__( self ):
    return '{}+x'.format( numeric.fstr(self.offset) )

class Scale( TransformItem ):

  def __init__( self, linear, offset ):
    assert linear.ndim == 0 and offset.ndim == 1
    self.linear = linear
    self.offset = offset
    self.isflipped = linear < 0 and len(offset)%2 == 1
    TransformItem.__init__( self, offset.shape[0], offset.shape[0] )

  def apply( self, points ):
    return self.linear * points + self.offset

  @property
  def det( self ):
    return self.linear**self.todims

  def __str__( self ):
    return '{}+{}*x'.format( numeric.fstr(self.offset), numeric.fstr(self.linear) )

class Matrix( TransformItem ):

  def __init__( self, linear, offset ):
    self.linear = linear
    self.offset = offset
    assert linear.ndim == 2 and offset.shape == linear.shape[:1]
    TransformItem.__init__( self, linear.shape[0], linear.shape[1] )

  def apply( self, points ):
    assert points.shape[-1] == self.fromdims
    return numpy.dot( points, self.linear.T ) + self.offset

  def __str__( self ):
    return '{}{}{}'.format( '~' if self.isflipped else '', numeric.fstr(self.offset), ''.join( '+{}*x{}'.format( numeric.fstr(v), i ) for i, v in enumerate(self.linear.T) ) )

class Square( Matrix ):

  def __init__( self, linear, offset ):
    Matrix.__init__( self, linear, offset )
    assert self.fromdims == self.todims

  @property
  def isflipped( self ):
    return self.det < 0

  @cache.property
  def det( self ):
    return numeric.det_exact( self.linear )

class Updim( Matrix ):

  def __init__( self, linear, offset, isflipped ):
    assert isflipped in (True,False)
    self.isflipped = isflipped
    Matrix.__init__( self, linear, offset )
    assert self.todims > self.fromdims

  @cache.property
  def ext( self ):
    ext = numeric.ext( self.linear )
    return -ext if self.isflipped else ext

  @property
  def flipped( self ):
    return Updim( self.linear, self.offset, not self.isflipped )

class Bifurcate( TransformItem ):
  'bifurcate'

  def __init__( self, trans1, trans2 ):
    assert trans1.fromdims == trans2.fromdims
    self.trans1 = trans1
    self.trans2 = trans2
    TransformItem.__init__( self, todims=trans1.todims if trans1.todims == trans2.todims else None, fromdims=trans1.fromdims )

  def apply( self, points ):
    return [ self.trans1.apply(points), self.trans2.apply(points) ]

class Slice( Matrix ):
  'slice'

  def __init__( self, i1, i2, fromdims ):
    todims = i2-i1
    assert 0 <= todims <= fromdims
    self.s = slice(i1,i2)
    self.isflipped = False
    Matrix.__init__( self, numpy.eye(fromdims)[self.s], numpy.zeros(todims) )

  def apply( self, points ):
    return points[:,self.s]

class VertexTransform( TransformItem ):

  def __init__( self, fromdims ):
    TransformItem.__init__( self, None, fromdims )
    self.isflipped = False

class MapTrans( VertexTransform ):

  def __init__( self, linear, offset, vertices ):
    assert len(linear) == len(offset) == len(vertices)
    self.vertices, self.linear, self.offset = map( numpy.array, zip( *sorted( zip( vertices, linear, offset ) ) ) ) # sort vertices
    VertexTransform.__init__( self, self.linear.shape[1] )

  def apply( self, points ):
    barycentric = numpy.dot( points, self.linear.T ) + self.offset
    return tuple( tuple( (v,float(c)) for v, c in zip( self.vertices, coord ) if c ) for coord in barycentric )

  def __str__( self ):
    return ','.join( str(v) for v in self.vertices )

class RootTrans( VertexTransform ):

  def __init__( self, name, shape ):
    VertexTransform.__init__( self, len(shape) )
    self.I, = numpy.where( shape )
    self.w = numpy.take( shape, self.I )
    self.name = name

  def apply( self, coords ):
    coords = numpy.asarray(coords)
    assert coords.ndim == 2
    if self.I.size:
      coords = coords.copy()
      coords[:,self.I] %= self.w
    return tuple( self.name + str(c) for c in coords.tolist() )

  def __str__( self ):
    return repr( self.name + '[*]' )

class RootTransEdges( VertexTransform ):

  def __init__( self, name, shape ):
    VertexTransform.__init__( self, len(shape) )
    self.shape = shape
    assert isinstance( name, numpy.ndarray )
    assert name.shape == (3,)*len(shape)
    self.name = name.copy()

  def apply( self, coords ):
    assert coords.ndim == 2
    labels = []
    for coord in coords.T.frac.T:
      right = (coord[:,1]==1) & (coord[:,0]==self.shape)
      left = coord[:,0]==0
      where = (1+right)-left
      s = self.name[tuple(where)] + '[%s]' % ','.join( str(n) if d == 1 else '%d/%d' % (n,d) for n, d in coord[where==1] )
      labels.append( s )
    return labels

  def __str__( self ):
    return repr( ','.join(self.name.flat)+'*' )


## CONSTRUCTORS

def affine( linear, offset, denom=1, isflipped=None ):
  r_offset = numpy.asarray( offset, dtype=float ) / denom
  r_linear = numpy.asarray( linear, dtype=float ) / denom
  n, = r_offset.shape
  if r_linear.ndim == 2:
    assert r_linear.shape[0] == n
    if r_linear.shape[1] != n:
      trans = Updim( r_linear, r_offset, isflipped )
    elif n == 0:
      trans = Shift( r_offset )
    elif n == 1 or r_linear[0,-1] == 0 and numpy.all( r_linear == r_linear[0,0] * numpy.eye(n) ):
      trans = Scale( r_linear[0,0], r_offset ) if r_linear[0,0] != 1 else Shift( r_offset )
    else:
      trans = Square( r_linear, r_offset )
  else:
    assert r_linear.ndim == 0
    trans = Scale( r_linear, r_offset ) if r_linear != 1 else Shift( r_offset )
  if isflipped is not None:
    assert trans.isflipped == isflipped
  return CanonicalTransformChain( [trans] )

def simplex( coords, isflipped=None ):
  coords = numpy.asarray(coords)
  offset = coords[0]
  return affine( (coords[1:]-offset).T, offset, isflipped=isflipped )

def roottrans( name, shape ):
  return CanonicalTransformChain(( RootTrans( name, shape ), ))

def roottransedges( name, shape ):
  return CanonicalTransformChain(( RootTransEdges( name, shape ), ))

def maptrans( linear, offset, vertices ):
  return CanonicalTransformChain(( MapTrans( linear, offset, vertices ), ))

def equivalent( trans1, trans2 ):
  trans1 = TransformChain( trans1 )
  trans2 = TransformChain( trans2 )
  if trans1 == trans2:
    return True
  while trans1 and trans2 and trans1[0] == trans2[0]:
    trans1 = trans1[1:]
    trans2 = trans2[1:]
  return numpy.all( fulllinear(trans1) == fulllinear(trans2) ) and numpy.all( offset(trans1) == offset(trans2) )


## INSTANCES

identity = CanonicalTransformChain()

def solve( T1, T2 ): # T1 << x == T2
  assert isinstance( T1, TransformChain )
  assert isinstance( T2, TransformChain )
  while T1 and T2 and T1[0] == T2[0]:
    T1 = T1[1:]
    T2 = T2[1:]
  if not T1:
    return TransformChain(T2)
  # A1 * ( Ax * xi + bx ) + b1 == A2 * xi + b2 => A1 * Ax = A2, A1 * bx + b1 = b2
  Ax, bx = numeric.solve_exact( linear(T1), linear(T2), offset(T2) - offset(T1) )
  return affine( Ax, bx )

def tensor( trans1, trans2 ):
  if not trans1 and not trans2:
    return identity
  return affine( trans1.linear if trans1.linear.ndim == 0 and trans2.linear.ndim == 0 and trans1.linear == trans2.linear
            else numeric.blockdiag([ trans1.linear, trans2.linear ]), numpy.concatenate([ trans1.offset, trans2.offset ]) )

def isflipped( chain ):
  return sum( trans.isflipped for trans in chain ) % 2 == 1

def linear( chain ):
  linear = numpy.array( 1. )
  for trans in chain:
    linear = numpy.dot( linear, trans.linear ) if linear.ndim and trans.linear.ndim \
        else linear * trans.linear
  return linear

def fulllinear( chain ):
  scale = 1
  linear = numpy.eye( chain[-1].fromdims )
  for trans in reversed(chain):
    if trans.linear.ndim == 0:
      scale *= trans.linear
    else:
      linear = numpy.dot( trans.linear, linear )
      if trans.todims > trans.fromdims:
        linear = numpy.concatenate( [ linear, trans.ext[:,_] ], axis=1 )
  return linear * scale

def linearfrom( chain, ndims ):
  if chain and ndims < chain[-1].fromdims:
    for i in reversed(range(len(chain))):
      if chain[i].todims == ndims:
        chain = chain[:i]
        break
    else:
      raise Exception( 'failed to find {}D coordinate system'.format(ndims) )
  if not chain:
    return numpy.eye( ndims )
  linear = fulllinear( chain )
  n, m = linear.shape
  if m >= ndims:
    return linear[:,:ndims]
  return numpy.concatenate( [ linear, numpy.zeros((n,ndims-m)) ], axis=1 )

def apply( chain, points ):
  for trans in reversed(chain):
    points = trans.apply( points )
  return points

def offset( chain ):
  offset = chain[-1].offset
  for trans in chain[-2::-1]:
    offset = trans.apply( offset )
  return offset

def slicetrans( i1, i2, n ):
  return CanonicalTransformChain( [ Slice(i1,i2,n) ] )

def stack( trans1, trans2 ):
  fromdims = trans1.fromdims + trans2.fromdims
  return bifurcate( trans1.canonical << slicetrans(0,trans1.fromdims,fromdims), trans2.canonical << slicetrans(trans1.fromdims,fromdims,fromdims) )

def bifurcate( trans1, trans2 ):
  return CanonicalTransformChain([ Bifurcate( trans1, trans2 ) ])

def invapply( trans, points ):
  A = linear(trans)
  b = points - offset(trans)
  return b / A if isinstance(A,float) else numpy.linalg.solve( A, b )

# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
