# Copyright 2024 The PySCF Developers. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
