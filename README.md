This repo repackges the DFT-D3 (https://github.com/dftd3/simple-dftd3) and
DFT-D4 (https://github.com/dftd4/dftd4) for PySCF. OpenMP for the two packages
are disabled in this repo due to its potential conflicts to PyTorch.
This is the main difference to the multi-threading versions of dftd3 and dftd4
released on PyPI.

```
pip install pyscf-dispersion
```

If PyTorch is not used in your program, you can still install the
multi-threading releases through `pip install dftd3 dftd4`
