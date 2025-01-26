import numpy as np
import pytest
import pyscf
from pyscf.dispersion.dftd4 import DFTD4Dispersion

vitamin = """
C                 -0.07551087    1.68127663   -0.10745193
O                  1.33621755    1.87147409   -0.39326987
C                  1.67074668    2.95729545    0.49387976
C                  0.41740763    3.77281969    0.78495878
C                 -0.60481480    3.07572636    0.28906224
H                 -0.19316298    1.01922455    0.72486113
O                  0.35092043    5.03413298    1.45545728
H                  0.42961487    5.74279041    0.81264173
O                 -1.95331750    3.53349874    0.15912025
H                 -2.55333895    2.78846397    0.23972698
O                  2.81976302    3.20110148    0.94542226
C                 -0.81772499    1.09230218   -1.32146482
H                 -0.70955636    1.74951833   -2.15888136
C                 -2.31163857    0.93420736   -0.98260166
H                 -2.72575463    1.89080093   -0.74107186
H                 -2.41980721    0.27699120   -0.14518512
O                 -0.26428017   -0.18613595   -1.64425697
H                 -0.72695910   -0.55328886   -2.40104423
O                 -3.00083741    0.38730252   -2.10989934
H                 -3.93210821    0.28874990   -1.89865997
"""

def test_d4():
    mol = pyscf.M(atom=vitamin)
    model = DFTD4Dispersion(mol, xc='r2scan3c')
    out = model.get_dispersion()

    print(out)

def test_d4_unknown_xc():
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1')
    with pytest.raises(RuntimeError):
        model = DFTD4Dispersion(mol, xc='wb97x-v')

def test_d4_energy():
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1')
    model = DFTD4Dispersion(mol, xc='WB97X')
    out = model.get_dispersion()
    assert abs(out['energy'] - -2.21334459527e-05) < 1e-10

def test_d4_gradients():
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1')
    model = DFTD4Dispersion(mol, xc='HF')
    out = model.get_dispersion(grad=True)
    assert abs(out['energy'] - -0.000967454204722) < 1e-10
    assert abs(out['gradient'][0,2] - 9.31972590827e-06) < 1e-10
    assert abs(out['virial'][2,2] - -1.76117295226e-05) < 1e-10

def test_d4_with_pbc():
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1', a=np.eye(3)*2)
    model = DFTD4Dispersion(mol, xc='WB97X')
    out = model.get_dispersion()
    assert abs(out['energy'] - -0.002715970438476524) < 1e-10
