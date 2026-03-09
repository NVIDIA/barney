 # Barney Native API: Supported Entities and Parameters

Barney can be driven using two different APIs: ANARI, or Barney
Native.  Both of those work primarily by creating certain 'actors'
(like Geometries, Volumes, Lights, Frame Buffers, etc) that the user
can 'parameterize' using certain `bnSetXyz()` calls to set their
internal state (e.g., setting the radiance and direction of a
directional light source).

<THIS FILE IS CURRENTLY INCOMPLETE>

## Non-Subtyped Objects

### `FrameBuffer` (`bnFrameBufferCreate()`)

- `int1 enableDenoising` : If set to 0, denoising will be skipped even
  if support for denoising is enabled during build, and offered by the
  chosen rtc backend. Any other value will turn denoising on
  (conditional upon the chosen backend offering it in the first place)

- `int1 enableUpscaling` : If set to non-zero, the frame buffer will
  render at half resolution and use the OptiX AI upscaler (UPSCALE2X)
  to produce full-resolution output. This significantly improves
  interactive performance with minimal quality loss. Requires an OptiX
  backend with OPTIX_VERSION >= 80000 and an NVIDIA RTX GPU. When
  enabled, denoising is automatically applied as part of the upscale
  pipeline.

## Lights (`bnLightCreate(<subtype>)`)

### Environment Map Light (`envmap`)

- `float scale = 1.f` : Scale factor applied to the value read from
  the `radiance[]` array.
- `float3 direction = (1,0,0)` : Direction that the env map should be
  oriented towards. This effectively 'rotates' the env-map around the
  `up` vector until direction points to the vertical center line of
  the env-map. If orthogonal to up direction will point to exactly the
  center pixel.
- `float3 up = (0,0,1)`: Specifies the logical (world-space) up vector
  for the specified environment map. Will point to the (top) pole of
  the sphere of directions.
- `Texture texture`: A (2D) Texture Object that specifies the texture
  that is wrapped around the environment map. Texel(0,0) is will appear
  in the *lower left* of the map.





