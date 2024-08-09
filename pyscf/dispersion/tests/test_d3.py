import numpy as np
import pytest
import pyscf
from pyscf.dispersion.dftd3 import DFTD3Dispersion

def test_d3_unknown_xc():
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1')
    with pytest.raises(RuntimeError):
        model = DFTD3Dispersion(mol, xc='wb97x-v')

def test_d3_energy():
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1')
    model = DFTD3Dispersion(mol, xc='WB97X')
    out = model.get_dispersion()
    assert abs(out['energy'] - -0.000289774198733) < 1e-10

def test_d3_gradients():
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1')
    model = DFTD3Dispersion(mol, xc='HF')
    out = model.get_dispersion(grad=True)
    assert abs(out['energy'] - -0.002682864283734) < 1e-10
    assert abs(out['gradient'][0, 2] - 0.0004553384490128) < 1e-10
    assert abs(out['virial'][2, 2] - -0.000860464962618) < 1e-10

def test_d3_with_pbc():
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1', a=np.eye(3)*2)
    model = DFTD3Dispersion(mol, xc='WB97X')
    out = model.get_dispersion()
    assert abs(out['energy'] - -0.0014506963767424985) < 1e-10
