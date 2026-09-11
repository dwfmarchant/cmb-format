# GemPy to CMB example

`gempy_to_cmb.py` builds a small synthetic model with two parallel dipping
contacts, evaluates its lithology at the cell centers of a `discretize`
`TensorMesh`, and writes the result to a CMB file. The file contains both a
categorical `lithology_id` model and a resistivity model, along with units,
lithology mapping, and provenance metadata. A cross-section PNG is generated
by default.

The example dependencies are optional and are intentionally not part of the
CMB package requirements. From a checkout, install the package and the demo
dependencies into an environment:

```bash
python -m pip install -e .
python -m pip install gempy discretize matplotlib
```

The current example was exercised with GemPy 2026.0.3, discretize 0.12.0,
and Matplotlib 3.11.1 on Python 3.12.

Run it from the repository root:

```bash
python examples/gempy_to_cmb.py
```

Generated files are placed in `examples/output/`, which is ignored by Git.
Use `--no-plot` when Matplotlib is unavailable, or choose another directory
with `--output-dir PATH`.
