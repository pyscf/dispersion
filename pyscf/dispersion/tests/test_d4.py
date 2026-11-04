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
    model = DFTD4Dispersion(mol, xc='WB97X-2008')
    out = model.get_dispersion()
    assert abs(out['energy'] - -2.21334459527e-05) < 1e-10

def test_wb97x_d4_energy():
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1')
    model = DFTD4Dispersion(mol, xc='WB97X')
    out = model.get_dispersion()
    assert abs(out['energy'] - -0.00027002) < 1e-8

def test_d4_gradients():
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1')
    model = DFTD4Dispersion(mol, xc='HF')
    out = model.get_dispersion(grad=True)
    assert abs(out['energy'] - -0.000967454204722) < 1e-10
    assert abs(out['gradient'][0,2] - 9.31972590827e-06) < 1e-10
    assert abs(out['virial'][2,2] - -1.76117295226e-05) < 1e-10

def test_d4_with_pbc():
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1', a=np.eye(3)*2)
    model = DFTD4Dispersion(mol, xc='WB97X-2008')
    out = model.get_dispersion()
    assert abs(out['energy'] - -0.002715970438476524) < 1e-10

def test_d4s_energy():
    ''' Test copied from DFTD4
    '''
    mol = pyscf.M(
        atom="""
             Na  -1.855282634   3.586705153  -2.417637293
             H    4.401780235   0.023388444  -4.954577493
             O   -2.987060334   4.762520654   1.270433015
             H    0.799808860   1.411034556  -5.046553216
             F   -4.206474694   1.842757675   4.550380848
             H   -3.543561218  -3.188356651   1.462400217
             H    2.700321601   1.068184525  -1.732346503
             O    3.731140888  -2.070015433   2.231609376
             N   -1.753068192   0.359514171   1.053234061
             H    5.417557885  -1.578818300   1.753940027
             H   -2.234628682  -2.138565050   4.109222857
             Cl   1.015658662  -3.219521545  -3.360509630
             B    2.421192557   0.266264350  -3.918624743
             B   -3.025260988   2.536678890   2.316649847
             N   -2.004389486  -2.292351369   2.197828073
             Al   1.122265541  -1.369420070   0.484550554
             """
    )
    model = DFTD4Dispersion(mol, xc="TPSS", version="d4s")
    out = model.get_dispersion()
    assert abs(out['energy'] - -0.016049411775539424) < 1.0e-7

def test_d4s_gradient():
    ''' Test copied from DFTD4
    '''
    mol = pyscf.M(
        atom="""
             H   -1.795376258  -3.778664229  -1.078835583
             S   -2.682788333   0.388926662   1.662148652
             B    0.114846497   1.488579332   3.656603965
             O   -1.079988795  -0.162591216  -4.557030658
             Mg   0.603028329   4.088161496  -0.025893731
             H   -1.225340893  -1.799813824  -3.707731733
             H   -1.334609820  -4.248190824   2.727919027
             H   -0.162780825   2.412679941   5.690306951
             Si   2.878024440  -0.331205250   1.883113735
             H    0.684893279   0.327902040  -4.205476937
             B   -1.209197735  -2.872537625   0.940642042
             Li  -3.255726045   2.212410929  -2.867155493
             F   -1.831474682   5.205272937  -2.269762706
             H    4.908858657  -1.925765619   2.990699194
             H    1.268062422  -2.604093417   0.551628052
             S    4.119569763   1.598928667  -1.391174777
             """,
        spin=1
    )
    ref = np.array(
        [
            [-1.04361222e-04, -1.65054791e-04, -1.36662175e-04],
            [-1.41500522e-03, +1.89282651e-04, +2.16639105e-04],
            [-1.18067839e-04, +4.50543787e-04, +1.50087553e-03],
            [+3.37690080e-04, -4.10348598e-04, -3.02311767e-04],
            [+4.39892308e-04, +1.54862493e-03, +1.33655085e-04],
            [+1.31259180e-06, -7.51721105e-05, -1.39848135e-04],
            [-4.61111364e-05, -1.65382677e-04, +1.81820530e-04],
            [-1.94292825e-05, +7.21791149e-05, +1.79879351e-04],
            [+1.14226323e-03, -6.08455689e-04, +6.24007890e-04],
            [+6.95738570e-05, -1.86718359e-05, -1.25837081e-04],
            [-1.66091884e-04, -1.03519307e-03, -1.71797180e-04],
            [-1.29925668e-03, +6.18658801e-05, -6.30138324e-04],
            [-1.58991399e-04, +5.73306273e-04, -2.35799582e-04],
            [+2.90056077e-04, -2.14985916e-04, +1.62430848e-04],
            [+6.43808246e-05, -3.35585457e-04, -2.45131168e-04],
            [+9.82145702e-04, +1.33047503e-04, -1.01178292e-03],
        ]
    )

    model = DFTD4Dispersion(mol, xc="BLYP", version="d4s")
    out = model.get_dispersion(grad=True)
    assert np.linalg.norm(out['gradient'] - ref) < 1.0e-7

def test_gradient_tpss_d4_charge():
    ''' Test copied from DFTD4 python/dftd4/test_qcschema.py test_gradient_tpss_d4
    '''
    mol = pyscf.M(
        atom="""
             O  +4.877023733 -3.909030492 +1.796260143
             C  +6.112318716 -2.778558610 +0.091330457
             C  +7.360520527 -4.445334728 -1.932830640
             F  +7.978801077 -6.767751279 -1.031771494
             O  +6.374499300 -0.460299457 -0.213142194
             F  +5.637581753 -4.819746139 -3.831249370
             H  +9.040657008 -3.585225944 -2.750722946
             """,
        charge=-1,
        unit='Bohr'
    )

    # Reference gradient from dftd4 test
    ref = np.array(
        [
            [-1.47959449e-04, +2.95411758e-05, +2.69548700e-04],
            [-9.35127114e-06, +2.37170201e-05, +2.01896552e-05],
            [+4.90329914e-05, -9.08636769e-06, -4.75377709e-05],
            [+1.28728759e-04, -1.98165729e-04, -5.63375526e-05],
            [-1.02103452e-05, +3.04524759e-04, +9.14084679e-05],
            [-4.27301731e-06, -1.00071311e-04, -2.42775990e-04],
            [-5.96766712e-06, -5.04595462e-05, -3.44955102e-05],
        ]
    )

    model = DFTD4Dispersion(mol, xc="TPSS", version="d4")

    out = model.get_dispersion(grad=True)
    assert np.linalg.norm(out['gradient'] - ref) < 1.0e-7


@pytest.mark.parametrize('distance,ref,switch', [
    (3.0, -0.00018238733798327627, 1.0),
    (4.0, -0.00017311086359280502, 0.5),
    (4.5, -0.0001623051934246008, 0.103515625),
    (5.0, -0.00014578629180959712, 0.0),
    (6.0, -9.945813723253093e-05, 0.0),
])
def test_d4_cutoff(distance, ref, switch):
    # Energies from dftd4 4.3.1 CLI: --func pbe, H2 coordinates in Bohr.
    mol = pyscf.M(atom=[('H', (0, 0, 0)), ('H', (0, 0, distance))], unit='Bohr')
    model = DFTD4Dispersion(mol, 'PBE')
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
        other = DFTD4Dispersion(mol.set_geom_(displaced, unit='Bohr', inplace=False), 'PBE')
        other.set_realspace_cutoff(5, 5, 10, width2=2, width3=2)
        energies.append(other.get_dispersion()['energy'])
    grad = model.get_dispersion(grad=True)['gradient'][1,2]
    assert abs(grad - (energies[1]-energies[0])/2e-6) < 1e-7


@pytest.mark.parametrize('dimension', [0, 1, 2, 3])
def test_d4_periodic_flags(dimension):
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1', a=np.eye(3)*4, dimension=dimension,
                  low_dim_ft_type='inf_vacuum')
    model = DFTD4Dispersion(mol, 'PBE')
    ref = model.get_dispersion()['energy']
    for axis in range(dimension):
        coords = mol.atom_coords()
        coords[1,axis] += mol.lattice_vectors()[axis,axis]
        shifted = DFTD4Dispersion(mol.set_geom_(coords, unit='Bohr', inplace=False), 'PBE')
        out = shifted.get_dispersion()['energy']
        assert abs(out - ref) < 1e-10
    if dimension == 0:
        molecule = pyscf.M(atom=mol.atom, basis=mol.basis)
        assert abs(ref - DFTD4Dispersion(molecule, 'PBE').get_dispersion()['energy']) < 1e-12


def test_d4_delete():
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1')
    model = DFTD4Dispersion(mol, 'PBE')
    model.__del__()
    assert not model._param
    assert not model._disp
    assert not model._mol
