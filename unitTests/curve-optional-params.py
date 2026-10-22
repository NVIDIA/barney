# Regression test: 'curve' geometry without its optional
# 'primitive.index' and 'vertex.radius' parameters.
#
# Without 'primitive.index' every vertex except the last one starts a
# segment; without 'vertex.radius' the global 'radius' applies. Both
# used to crash barney: building the geometry dereferenced the missing
# index array, and computing the world bounds the missing radius array.

import math
import numpy as np
import pynari as anari

fb_size = (64, 64)
radius = .2

device = anari.newDevice('default')

# a polyline from x=-1 to x=+1, made of segments (0,1) and (1,2)
positions = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=np.float32)

geom = device.newGeometry('curve')
geom.setParameter('vertex.position', anari.ARRAY1D,
                  device.newArray1D(anari.FLOAT32_VEC3, positions))
geom.setParameter('radius', anari.FLOAT32, radius)
geom.commitParameters()

material = device.newMaterial('matte')
material.commitParameters()

surface = device.newSurface()
surface.setParameter('geometry', anari.GEOMETRY, geom)
surface.setParameter('material', anari.MATERIAL, material)
surface.commitParameters()

world = device.newWorld()
world.setParameterArray1D('surface', anari.SURFACE, [surface])
world.commitParameters()

bounds = world.getBounds()
expected = (-1 - radius, -radius, -radius, 1 + radius, radius, radius)
assert all(abs(b - e) < 1e-5 for b, e in zip(bounds, expected)), \
    f'world bounds are {bounds}, expected {expected}'

# looking down -z from 5 units away; the view spans [-2,2] in x and y
camera = device.newCamera('perspective')
camera.setParameter('position', anari.FLOAT32_VEC3, (0., 0., 5.))
camera.setParameter('direction', anari.FLOAT32_VEC3, (0., 0., -1.))
camera.setParameter('up', anari.FLOAT32_VEC3, (0., 1., 0.))
camera.setParameter('fovy', anari.FLOAT32, 2. * math.atan(2. / 5.))
camera.setParameter('aspect', anari.FLOAT32, 1.)
camera.commitParameters()

renderer = device.newRenderer('default')
renderer.setParameter('background', anari.FLOAT32_VEC4, (0., 0., 0., 1.))
renderer.commitParameters()

frame = device.newFrame()
frame.setParameter('size', anari.uint2, fb_size)
frame.setParameter('channel.color', anari.DATA_TYPE, anari.UFIXED8_RGBA_SRGB)
frame.setParameter('renderer', anari.RENDERER, renderer)
frame.setParameter('camera', anari.CAMERA, camera)
frame.setParameter('world', anari.WORLD, world)
frame.commitParameters()

frame.render()
pixels = np.array(frame.get('channel.color'))

def covered(x, y):
    px = int((x / 4. + .5) * fb_size[0])
    py = int((y / 4. + .5) * fb_size[1])
    return int(pixels[py][px][:3].sum()) > 0

assert covered(-.5, 0.), 'segment (0,1) was not rendered'
assert covered(.5, 0.), 'segment (1,2) was not rendered'
assert not covered(0., 1.), 'something was rendered away from the curve'
print('@pynari: curve without primitive.index and vertex.radius: ok')
