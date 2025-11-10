# ast-net


⚠️ Warning: This is an experimental project!

This repository contains `ast-net`, a deep learning network that classifies bacterial
species related to sepsis. This repository is currently *unstable* and will contain
non-final code and experiments.

## Install `ast-net` from souce

To install ast-net from source first install the Rust toolchain from [rust-lang.org](https://rust-lang.org/tools/install/).
Next create an environment (we recommend using `uv`) with the `maturin` development tool. This can be easily done with the
`uv` tool and this repository's `pyproject.toml`.

```bash
$ cd ast-net
$ uv sync
```

This will create the environment for you with maturin. Next install Rust library with:

```bash
$ maturin develop --release
```

## Example

You can run the following stardist example:

```python
import imagej
import ast_net
import numpy as np

# initialize imagej
ij = imagej.init(mode = "interactive")

# load the data and convert it to float32
# the data MUST be 512x512
data = ij.io().open("/path/to/stardist_test_data.tif")
narr = ij.py.to_xarray(data).data
narr = narr.astype(np.float32)

# run stardist 2D inference with wgpu backend
result = ast_net.stardist.stardist_2d(narr)
ij.py.show(result)
