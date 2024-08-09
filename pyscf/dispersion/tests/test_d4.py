import numpy as np
import pytest
import pyscf
from pyscf.dispersion.dftd4 import DFTD4Dispersion

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
