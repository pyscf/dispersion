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
from pyscf.dispersion.gcp import GCP

def test_gcp_unknown_xc():
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1')
    with pytest.raises(RuntimeError):
        model = GCP(mol, method='xx')
        out = model.get_counterpoise()

def test_b973c_energy():
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1')
    model = GCP(mol, method='b973c')
    out = model.get_counterpoise(grad=True)
    assert abs(out['energy'] - -0.000818489452022319) < 1e-8
    assert abs(out['gradient'][0,2] - -0.0019847223817803105) < 1e-8
    assert abs(out['virial'][2,2] - 0.0037505817348592453) < 1e-8

def test_r2scan3c_energy():
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1')
    model = GCP(mol, method='r2scan3c')
    out = model.get_counterpoise(grad=True)
    assert abs(out['energy'] - 0.0002982079859909563) < 1e-8
    assert abs(out['gradient'][0,2] - -0.0004954338396206401) < 1e-8
    assert abs(out['virial'][2,2] - 0.0009362342697247007) < 1e-8

# wb97x-3c does not require GCP
def test_wb97x3c_energy():
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1')
    model = GCP(mol, method='wb97x3c')
    out = model.get_counterpoise(grad=True)
    for value in out.values():
        assert abs(value).max() < 1e-8


@pytest.mark.parametrize('method', ['b973c', 'r2scan3c'])
def test_gcp_hessian(method):
    mol = pyscf.M(atom='O 0 0 0; H 0 0 1; H 0 1 0')
    model = GCP(mol, method=method)
    out = model.get_hessian()
    assert abs(out['energy'] - model.get_counterpoise()['energy']) < 1e-12
    hess = out['hessian']
    assert abs(hess - hess.T).max() < 1e-12
    coords = mol.atom_coords()
    numerical = np.zeros_like(hess)
    for i in range(mol.natm*3):
        shift = np.zeros_like(coords)
        shift.flat[i] = 1e-6
        plus = GCP(mol.set_geom_(coords+shift, unit='Bohr', inplace=False), method=method)
        minus = GCP(mol.set_geom_(coords-shift, unit='Bohr', inplace=False), method=method)
        numerical[i] = ((plus.get_counterpoise(grad=True)['gradient'] -
                         minus.get_counterpoise(grad=True)['gradient']) / 2e-6).ravel()
    assert abs(hess - numerical).max() < 1e-7


@pytest.mark.parametrize('distance,ref', [
    (3.0, -5.543447726619787e-05),
    (4.0, -4.905424695456383e-06),
    (4.5, -1.45923400427623e-06),
    (5.0, -4.340834915289495e-07),
    (6.0, -3.84122658722076e-08),
])
def test_gcp_cutoff(distance, ref):
    # Energies from s-dftd3 1.7.0 CLI: gcp --level b973c, H2 coordinates in Bohr.
    mol = pyscf.M(atom=[('H', (0, 0, 0)), ('H', (0, 0, distance))], unit='Bohr')
    model = GCP(mol, method='b973c')
    baseline = model.get_counterpoise(grad=True)
    assert abs(baseline['energy'] - ref) < 1e-12
    model.set_realspace_cutoff(2.5, 2.5)
    out = model.get_counterpoise(grad=True)
    for value in out.values():
        assert abs(value).max() < 1e-15
    model.set_realspace_cutoff(5, 5)
    out = model.get_counterpoise(grad=True)
    # gCP uses a hard cutoff and includes pairs at the cutoff distance.
    switch = float(distance <= 5)
    assert abs(out['energy'] - switch*ref) < 1e-12
    for key in baseline:
        assert abs(out[key] - switch*baseline[key]).max() < 1e-12
    # Skip the gradient check at the discontinuous hard-cutoff boundary.
    if distance != 5:
        coords = mol.atom_coords()
        energies = []
        for shift in [-1e-6, 1e-6]:
            displaced = coords.copy()
            displaced[1,2] += shift
            other = GCP(mol.set_geom_(displaced, unit='Bohr', inplace=False), method='b973c')
            other.set_realspace_cutoff(5, 5)
            energies.append(other.get_counterpoise()['energy'])
        grad = out['gradient'][1,2]
        assert abs(grad - (energies[1]-energies[0])/2e-6) < 1e-7


@pytest.mark.parametrize('dimension', [0, 1, 2, 3])
def test_gcp_periodic_flags(dimension):
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1', a=np.eye(3)*4, dimension=dimension,
                  low_dim_ft_type='inf_vacuum')
    model = GCP(mol, method='b973c')
    ref = model.get_counterpoise()['energy']
    for axis in range(dimension):
        coords = mol.atom_coords()
        coords[1,axis] += mol.lattice_vectors()[axis,axis]
        shifted = GCP(mol.set_geom_(coords, unit='Bohr', inplace=False), method='b973c')
        out = shifted.get_counterpoise()['energy']
        assert abs(out - ref) < 1e-10
    if dimension == 0:
        molecule = pyscf.M(atom=mol.atom, basis=mol.basis)
        assert abs(ref - GCP(molecule, method='b973c').get_counterpoise()['energy']) < 1e-12


def test_gcp_delete():
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1')
    model = GCP(mol, method='b973c')
    model.__del__()
    assert not model._gcp
    assert not model._mol
