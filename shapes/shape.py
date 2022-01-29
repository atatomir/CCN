from email.errors import InvalidBase64LengthDefect
import numpy as np
import matplotlib.pyplot as plt

class Shape:
  def __init__(self):
    pass

  def inside(self, x, y):
    raise NotImplementedError 

  def __and__(self, other):
    return FunctionalShape(lambda x, y: self.inside(x, y) and other.inside(x, y))

  def __or__(self, other):
    return FunctionalShape(lambda x, y: self.inside(x, y) or other.inside(x, y))

  def __neg__(self):
    return FunctionalShape(lambda x, y: not self.inside(x, y))

  def find_borders(self, image):
    n, m = image.shape
    image = image[1:, 1:] + image[1:, :(m - 1)] + image[:(n - 1), 1:] + image[:(n - 1), :(m - 1)]
    result = np.zeros((n - 1, m - 1))
    result[(0 < image) & (image < 4)] = 1.
    return result

  def plot(self, ax, full=False):
    dots = np.arange(0., 1., 0.01)
    grid = [(x, y) for y in dots for x in dots]

    inside = [(1. if self.inside(x, y) else 0.) for (x, y) in grid]
    image = np.array(inside).reshape(len(dots), len(dots))
    if not full: 
      image = self.find_borders(image)

    ax.imshow(image, alpha=image * 0.6, cmap='winter', interpolation='nearest', origin='lower', extent=(0., 1., 0., 1.))


class FunctionalShape(Shape):
  def __init__(self, inside_check):
    self.inside_check = inside_check 

  def inside(self, x, y):
    return self.inside_check(x, y) 


class Rectangle(Shape):
  def __init__(self, x0, x1, y0, y1):
    self.x = (x0, x1)
    self.y = (y0, y1)

  def inside(self, x, y):
    x0, x1 = self.x 
    y0, y1 = self.y 
    return x0 <= x and x <= x1 and y0 <= y and y <= y1

class HalfPlane(Shape):
  def __init__(self, a, b, c):
    self.consts = (a, b, c)

  def inside(self, x, y):
    a, b, c = self.consts
    return a * x + b * y + c <= 0.

class Circle(Shape):
  def __init__(self, x, y, r):
    self.centre = (x, y)
    self.r = r

  def inside(self, x, y):
    (cx, cy) = self.centre 
    return (x - cx) * (x - cx) + (y - cy) * (y - cy) <= self.r * self.r

def test_shapes():
  rect0 = Rectangle(0.5, 0.7, 0.7, 0.9)
  rect1 = Rectangle(0.2, 0.6, 0.2, 0.8)
  half = HalfPlane(1, -1, 0)
  circle = Circle(0.7, 0.3, 0.2)
  shape = ((rect0 | rect1) & half) | circle 

  assert shape.inside(0.38, 0.47)
  assert shape.inside(0.64, 0.85)
  assert shape.inside(0.77, 0.37)
  assert not shape.inside(0.43, 0.37)
  assert not shape.inside(0.63, 0.66)
  assert not shape.inside(0.55, 0.13)





