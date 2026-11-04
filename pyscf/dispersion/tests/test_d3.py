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

@pytest.mark.parametrize('dimension,energy', [
    (1, -0.0014506963767424985),
    (3, -0.010868454099032226), # s-dftd3 1.7.0 CLI, 3D periodic input
])
def test_d3_with_pbc(dimension, energy):
    # The old single-bool pointer inadvertently selected one-dimensional PBC.
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1', a=np.eye(3)*2,
                  dimension=dimension, low_dim_ft_type='inf_vacuum')
    model = DFTD3Dispersion(mol, xc='WB97X')
    out = model.get_dispersion()
    assert abs(out['energy'] - energy) < 1e-10

@pytest.mark.parametrize('version', ['d3bj', 'd3zero', 'd3cso'])
@pytest.mark.parametrize('atm', [False, True])
def test_d3_hessian(version, atm):
    mol = pyscf.M(atom='O 0 0 0; H 0 0 1; H 0 1 0')
    model = DFTD3Dispersion(mol, 'PBE', version=version, atm=atm)
    out = model.get_hessian()
    assert abs(out['energy'] - model.get_dispersion()['energy']) < 1e-12
    hess = out['hessian']
    assert abs(hess - hess.T).max() < 1e-12
    coords = mol.atom_coords()
    numerical = np.zeros_like(hess)
    for i in range(mol.natm*3):
        shift = np.zeros_like(coords)
        shift.flat[i] = 1e-6
        plus = DFTD3Dispersion(mol.set_geom_(coords+shift, unit='Bohr', inplace=False),
                              'PBE', version=version, atm=atm)
        minus = DFTD3Dispersion(mol.set_geom_(coords-shift, unit='Bohr', inplace=False),
                               'PBE', version=version, atm=atm)
        numerical[i] = ((plus.get_dispersion(grad=True)['gradient'] -
                         minus.get_dispersion(grad=True)['gradient']) / 2e-6).ravel()
    assert abs(hess - numerical).max() < 1e-7


@pytest.mark.parametrize('distance,ref,switch', [
    (3.0, -0.00021263764907232066, 1.0),
    (4.0, -0.00019917983832575575, 0.5),
    (4.5, -0.00018393983784019185, 0.103515625),
    (5.0, -0.00016152233069943950, 0.0),
    (6.0, -0.00010387332656142640, 0.0),
])
def test_d3_cutoff(distance, ref, switch):
    # Energies from s-dftd3 1.7.0 CLI: --bj pbe, H2 coordinates in Bohr.
    mol = pyscf.M(atom=[('H', (0, 0, 0)), ('H', (0, 0, distance))], unit='Bohr')
    model = DFTD3Dispersion(mol, 'PBE')
    assert abs(model.get_dispersion()['energy'] - ref) < 1e-12
    model.set_realspace_cutoff(3, 3, 10, width2=1, width3=1)
    out = model.get_dispersion(grad=True)
    for value in out.values():
        assert abs(value).max() < 1e-15
    model.set_realspace_cutoff(5, 5, 10, width2=2, width3=2)
    # S(x) = 10*x**3 - 15*x**4 + 6*x**5, x = (5-distance)/2, clipped to [0,1].
    assert abs(model.get_dispersion()['energy'] - switch*ref) < 1e-12
    coords = mol.atom_coords()
    energies = []
    for shift in [-1e-6, 1e-6]:
        displaced = coords.copy()
        displaced[1,2] += shift
        other = DFTD3Dispersion(mol.set_geom_(displaced, unit='Bohr', inplace=False), 'PBE')
        other.set_realspace_cutoff(5, 5, 10, width2=2, width3=2)
        energies.append(other.get_dispersion()['energy'])
    grad = model.get_dispersion(grad=True)['gradient'][1,2]
    assert abs(grad - (energies[1]-energies[0])/2e-6) < 1e-7


def test_d3_ewald():
    mol = pyscf.M(atom='C .4 .8 1.2; O 4.4 2.8 3.6', unit='Bohr', a=np.eye(3)*8)
    model = DFTD3Dispersion(mol, 'PBE')
    model.set_realspace_cutoff(400, 40, 40)
    ref = model.get_dispersion()['energy']
    model.set_realspace_cutoff(40, 40, 40)
    model.set_ewald_summation(kcut=10, mesh=-1)
    direct = model.get_dispersion()['energy']
    model.set_ewald_summation(kcut=10)
    mesh = model.get_dispersion()['energy']
    assert abs(direct - ref) < 1e-7
    assert abs(mesh - direct) < 1e-8


@pytest.mark.parametrize('dimension', [0, 1, 2, 3])
def test_d3_periodic_flags(dimension):
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1', a=np.eye(3)*4, dimension=dimension,
                  low_dim_ft_type='inf_vacuum')
    model = DFTD3Dispersion(mol, 'PBE')
    ref = model.get_dispersion()['energy']
    for axis in range(dimension):
        coords = mol.atom_coords()
        coords[1,axis] += mol.lattice_vectors()[axis,axis]
        shifted = DFTD3Dispersion(mol.set_geom_(coords, unit='Bohr', inplace=False), 'PBE')
        out = shifted.get_dispersion()['energy']
        assert abs(out - ref) < 1e-10
    if dimension == 0:
        molecule = pyscf.M(atom=mol.atom, basis=mol.basis)
        assert abs(ref - DFTD3Dispersion(molecule, 'PBE').get_dispersion()['energy']) < 1e-12



def test_d3_delete():
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1')
    model = DFTD3Dispersion(mol, 'PBE')
    model.__del__()
    assert not model._param
    assert not model._disp
    assert not model._mol
