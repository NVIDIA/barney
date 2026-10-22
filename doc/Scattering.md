# Native scattering contract

The native renderer and packed BSDFs share the contracts below. The interface
lives in `barney/native/render/ShadingInteraction.h` and
`barney/native/packedBSDF/PackedBSDF.h`; integration and light selection live in
`barney/native/kernels/shadeRays.cu`.

## Coordinates and normal roles

Positions, directions, and both surface normals are world-space. `wo` points
from the hit toward the previous path vertex; evaluated or sampled `wi` points
from the hit toward the light or next vertex. BSDF directions are unit vectors.
“Shading” versus “geometric” describes a normal's role, not its coordinate space.

- The geometric normal determines front/back classification, medium sidedness,
  reflection/transmission support, and surface ray offsets. `ShadingInteraction::geometricNormal` is oriented
  toward `wo`. Reflection stays on its side; transmission crosses it.
- `ShadingInteraction::shadingNormal` is the original interpolated shading normal, before material normal
  maps and before orientation. Material maps perturb this shading frame, not
  the geometric normal. Base and clearcoat maps are resolved independently.
- `orientShadingNormal()` applies one shared sign, determined by `dot(geometricNormal,shadingNormal)`,
  to the original or mapped normal. Mapped lobes are not independently flipped
  toward `wo`. A lobe facing away may therefore have no support.
- Each lobe's shading normal determines its projection cosine. Clearcoat can
  have support and projection independent of the base lobe. Geometric normals
  still restrict which side a scattering event may reach.

Triangle hits carry geometric and interpolated normals independently. Analytic
primitives use the shading normal as the geometric fallback when no separate
normal is supplied. Ray normals are normalized before half-precision storage
and after unpacking. A zero geometric normal identifies a volume hit.

## Evaluation versus sampled throughput

`EvalRes` describes only the continuous part of scattering:

| Operation | `value` | `pdf` |
| --- | --- | --- |
| `eval(interaction, wi)` | Raw BSDF, or albedo-weighted phase function | Marginal solid-angle density |
| `evalProjected(interaction, wi)` | BSDF with each lobe's incident cosine included | Same density |
| `pdf(interaction, wi)` | Not returned | Same density |

Neither evaluation operation divides by the PDF. A phase function has no surface
projection, so its raw and projected values are identical. For layered surfaces,
projected evaluation is not necessarily raw evaluation times one shared cosine.

The density describes the complete sampling strategy, including lobe selection
and reflection/transmission branching, not just the lobe that produced a sample.
Unsupported directions return zero. Sampling may reject a direction; the density
is not renormalized to condition on acceptance.

`ScatterResult::weight` is instead the **complete radiance-throughput multiplier**.
For a continuous sample:

```
weight = evalProjected(interaction, dir).value / pdf(interaction, dir)
newThroughput = incomingThroughput * weight
```

The renderer must not apply another cosine, PDF division, or refractive-index
factor. BSDF weights own those factors, including the radiance-mode eta-squared
factor where applicable.

## Continuous and discrete measures

A continuous sample has `delta == false` and a positive finite solid-angle `pdf`.
A discrete sample has `delta == true` and `pdf == 0`; its weight already includes
compensation for its discrete event-selection probability. Zero here does not
mean zero probability mass, and infinity is not used as a delta-PDF sentinel.
Continuous evaluation and `pdf()` exclude discrete contributions.

`event` distinguishes reflection, transmission, and volume scattering. `type`
classifies the sampled lobe for policies such as diffuse-bounce limits; it is
not the probability measure. `eta` records incident/transmitted refractive index
and is one for reflection. Medium transitions follow transmission events, not
lobe type or PDF magnitude.

Ideal glass is discrete. Rough PBR and NVisii transmission are continuous and
can compete with environment sampling. Index-matched transmission collapses to
a discrete straight-through direction even when the material has roughness.

## Light sampling and MIS

`Light::Sample` separates selection probability from conditional direction
density. The renderer selects uniformly among individual quad, directional,
and point lights, plus one environment entry when present. The environment is
either a texture or constant ambient radiance.

For continuous directions, the estimator denominator is:

```
selectionProbability * conditionalSolidAnglePdf
```

For point and directional lights, `delta` is true and the denominator is just
`selectionProbability`. Quad samples convert uniform area density into solid
angle; sampled radiance does not contain that conversion.

The balance heuristic combines environment light sampling and continuous BSDF
sampling using their full densities, including environment selection probability.
Discrete events do not compete with continuous direction sampling. Quad lights
have no intersectable geometry in this implementation, so there is no competing
BSDF-hit estimator for them either. Both cases receive MIS weight one.

Environment CDF construction handles zero-energy rows and maps without dividing
by zero. Directional density uses the queried direction's latitude, matching the
sampling Jacobian rather than a texel-center approximation.

## Material and integrator changes accompanying this contract

The scattering unification also changes behavior; it is not only an API rename:

- Matte stores the supplied albedo, without the old extra 0.85 reflectance or
  pre-applied `1/pi` factors. Lambertian evaluation owns `albedo/pi`.
- Glass, NVisii, PBR, and phase sampling produce complete sample weights.
- PBR shares value/PDF reconstruction to keep sharp microfacet factors
  consistent, and evaluates rough transmission with its matching density.
- SimplePhase samples a unit sphere with the corrected transverse scale.
- The integrator applies sample weights directly and removes the old
  per-bounce throughput clamp.

These contracts do not by themselves establish physical correctness or replace
material-specific numerical and rendering tests.
