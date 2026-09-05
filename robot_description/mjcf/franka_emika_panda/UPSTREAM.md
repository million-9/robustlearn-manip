# Franka Emika Panda MuJoCo Model Provenance

Source repository:

https://github.com/google-deepmind/mujoco_menagerie

Upstream model directory:

franka_emika_panda

Vendored upstream commit:

`8161bba264d7fa7c99ca301e91e7fb44737676ad`

Primary robot model:

`panda.xml`

Reference scene:

`scene.xml`

License:

Apache-2.0. See `LICENSE` in this directory.

The upstream model is vendored into RobustLearn-Manip so the simulation does
not depend on an external checkout or an absolute filesystem path.

Only the standard MuJoCo model required by the flagship is vendored here.
MJX-specific variants are intentionally excluded.
