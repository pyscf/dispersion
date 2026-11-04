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

import os
import numpy as np
import ctypes
from pyscf import lib, gto
from pyscf.dispersion.dftd3 import libdftd3, _d3_p, error_check

libdftd3.dftd3_new_error.restype                   = _d3_p
libdftd3.dftd3_new_structure.restype               = _d3_p
libdftd3.dftd3_load_gcp_param.restype              = _d3_p
libdftd3.dftd3_check_error.restype                 = ctypes.c_int

# Somehow, GCP does not check the error for default method or basis
_white_list = {
    'b973c',
    'r2scan3c',
    'wb97x3c'
}

class GCP(lib.StreamObject):
    def __init__(self, mol, method=None, basis=None):
        self._gcp = None
        self._mol = None
        if method is not None and method.lower() not in _white_list:
            raise RuntimeError()

        coords = np.asarray(mol.atom_coords(), dtype=np.double, order='C')
        nuc_types = [gto.charge(mol.atom_symbol(ia))
                     for ia in range(mol.natm)]
        nuc_types = np.asarray(nuc_types, dtype=np.int32)
        self.natm = mol.natm
        if isinstance(mol, gto.Mole):
            lattice = lib.c_null_ptr()
            periodic = lib.c_null_ptr()
        else: # pbc.gto.Cell
            a = mol.lattice_vectors()
            lattice = a.ctypes
            periodic = (ctypes.c_bool*3)(*[i < mol.dimension for i in range(3)])

        err = libdftd3.dftd3_new_error()
        try:
            self._mol = libdftd3.dftd3_new_structure(
                err,
                ctypes.c_int(mol.natm),
                nuc_types.ctypes.data_as(ctypes.c_void_p),
                coords.ctypes.data_as(ctypes.c_void_p),
                lattice, periodic,
            )
            error_check(err)

            method_ptr = lib.c_null_ptr()
            basis_ptr = lib.c_null_ptr()
            if method is not None:
                method_ptr = method.encode()
            if basis is not None:
                basis_ptr = basis.encode()
        
            self._gcp = libdftd3.dftd3_load_gcp_param(
                err,
                self._mol,
                method_ptr,
                basis_ptr)
            error_check(err)
        finally:
            libdftd3.dftd3_delete_error(ctypes.byref(err))
        
    def __del__(self):
        if self._gcp:
            libdftd3.dftd3_delete_gcp(ctypes.byref(self._gcp))
        if self._mol:
            libdftd3.dftd3_delete_structure(ctypes.byref(self._mol))

    def set_realspace_cutoff(self, bas, srb):
        '''Set realspace cutoffs for gCP and short-range corrections in Bohr.

        Args:
            bas : float
                Cutoff radius for the geometric counterpoise correction to
                basis-set superposition error.
            srb : float
                Cutoff radius for the short-range bond correction to basis-set
                incompleteness, when included in the selected method.
        '''
        err = libdftd3.dftd3_new_error()
        try:
            libdftd3.dftd3_set_gcp_realspace_cutoff(
                err,
                self._gcp,
                ctypes.c_double(bas),
                ctypes.c_double(srb))
            error_check(err)
        finally:
            libdftd3.dftd3_delete_error(ctypes.byref(err))

    def get_hessian(self):
        '''Return counterpoise energy and analytical Cartesian Hessian.

        Returns:
            A dict with keys 'energy' (scalar array in Hartree) and 'hessian'
            ((3N,3N) array in Hartree/Bohr**2), where N is the number of atoms.
            Both Hessian axes follow the order x1, y1, z1, x2, y2, z2, ...
        '''
        _energy = np.array(0.0, dtype=np.double)
        _hessian = np.zeros((3*self.natm,3*self.natm))
        err = libdftd3.dftd3_new_error()
        try:
            libdftd3.dftd3_get_counterpoise_hessian(
                err,
                self._mol,
                self._gcp,
                _energy.ctypes.data_as(ctypes.c_void_p),
                _hessian.ctypes.data_as(ctypes.c_void_p))
            error_check(err)
        finally:
            libdftd3.dftd3_delete_error(ctypes.byref(err))
        return dict(energy=_energy, hessian=_hessian)

    def get_counterpoise(self, grad=False):
        res = {}
        _energy = np.array(0.0, dtype=np.double)
        _energy_str = _energy.ctypes.data_as(ctypes.c_void_p)
        if grad:
            _gradient = np.zeros((self.natm,3))
            _sigma = np.zeros((3,3))
            _gradient_str = _gradient.ctypes.data_as(ctypes.c_void_p)
            _sigma_str = _sigma.ctypes.data_as(ctypes.c_void_p)
        else:
            _gradient = None
            _sigma = None
            _gradient_str = lib.c_null_ptr()
            _sigma_str = lib.c_null_ptr()

        err = libdftd3.dftd3_new_error()
        try:
            libdftd3.dftd3_get_counterpoise(
                err,
                self._mol,
                self._gcp,
                _energy_str,
                _gradient_str,
                _sigma_str)
            error_check(err)

            res = dict(energy=_energy)
            if _gradient is not None:
                res.update(gradient=_gradient)
            if _sigma is not None:
                res.update(virial=_sigma)
        finally:
            libdftd3.dftd3_delete_error(ctypes.byref(err))

        return res
