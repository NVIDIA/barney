 # Barney ANARI API Layer: Supported Entities and Parameters

Barney can be driven using two different APIs: ANARI, or Barney
Native. While barney native is not ANARI compliant (and may or may not
offer features not exposed through ANARI), the ANARI API layer
basically follows the ANARI API Spec (see
https://registry.khronos.org/ANARI/specs/1.0/ANARI-1.0.html).

<THIS FILE IS CURRENTLY INCOMPLETE>

## Non-Subtyped Objects

### `Frame` (`anariNewFrame()`)

- `int1 enableDenoising` : If set to 0, denoising will be skipped even
  if support for denoising is enabled during build, and offered by the
  chosen rtc backend. Any other value will turn denoising on
  (conditional upon the chosen backend offering it in the first place)

- `int1 enableUpscaling` : If set to non-zero, renders at half
  resolution and uses the OptiX AI upscaler (UPSCALE2X) for
  full-resolution output. Requires OptiX 8+ and NVIDIA RTX GPU.

## Lights (`bnLightCreate(<subtype>)`)

### Environment Map/HDRI Light (`hdri`)

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
- `float3 radiance [][]`: 2D Array of radiance values. The
  `radiance[0][0]` value will appear in the *lower left* of the map.





