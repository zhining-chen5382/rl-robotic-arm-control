# Third-party components and source provenance

The SIASUN supervisor's permission to publish the related algorithm source and the project author's approval to include v30 best are recorded for this public source release. Team framework contributions remain acknowledged in [Contributions](docs/contributions.md).

No project-wide reuse license has been selected for the original project code or trained policy. Permission to publish a portfolio is distinct from granting all downstream reuse rights. Existing third-party licenses continue to apply to their components.

| Component | Repository location | Retained license / provenance |
|---|---|---|
| Three.js and selected loaders/controls | `code/static/vendor/three/` | [MIT license](code/static/vendor/three/LICENSE), retained verbatim |
| Panda URDF and meshes | `code/urdf/` | [Apache 2.0 text](code/urdf/LICENSE.txt); 30 files match the PyBullet 3.2.5 distribution and its RECORD hashes |
| Derived Panda MJCF | `code/mjx/assets/` | Converted robot description with fixed hand/finger transformations; conversion and repair source retained |
| v30 best | `code/models/` | One approved inference policy; [checkpoint metadata](code/models/README.md) |
| Python dependencies | Installed separately | Their upstream packages retain their licenses; they are not vendored here |

## Panda resource provenance

The Panda URDF, 28 mesh/material/texture files, and license text match `pybullet_data/franka_panda` in the installed **PyBullet 3.2.5** distribution byte for byte, including its package RECORD SHA-256 entries. The package metadata identifies [bulletphysics/bullet3](https://github.com/bulletphysics/bullet3) as its upstream project. [Per-file provenance](code/urdf/PROVENANCE.json) records this check without local computer paths. This identifies a distribution version; the exact upstream Git revision is not established. The resource-specific Apache 2.0 text is retained; it differs from the Python package's overall zlib license metadata.

An extra unverified Xacro file was excluded from the candidate. Runtime uses the expanded Panda URDF and the derived MJCF files. The derived MJCF transformations are documented by the retained conversion and hand/finger repair scripts.

The robot description and meshes are third-party assets with provenance and licenses documented above. No UR5 assets, company registration documents, reference templates, local Python environment, or historical checkpoint collection is included.

## Release modifications

The current application adds explicit start-state validation, actual-start obstacle protection, state-consistent scene replay/export, dynamic comparison labels, a deployment evaluation CLI, and an explicit training configuration argument. Historical source selections are frozen copies and labelled separately. The original research workspace is preserved outside this repository.
