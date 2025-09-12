"""Training-time augmentations, in two stages.

1. ``augment_raw`` acts on the raw sequence ``[T, L, 3]`` (image coordinates, before
   preprocessing): non-uniform time warping, frame dropping, landmark/hand dropping, noise.
2. ``spatial`` acts on the *normalised* sequence inside ``preprocess`` (body frame: origin at the
   shoulder centre, unit = shoulder width): rotation, anisotropic scaling and translation.

Spatial transforms are applied after normalisation on purpose: an image-space translation or
isotropic scaling would be cancelled exactly by the shoulder normalisation.
All functions take an explicit ``numpy.random.Generator`` and never modify their input.
"""

