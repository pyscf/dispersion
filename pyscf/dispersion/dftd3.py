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

libdftd3 = lib.load_library('libs-dftd3')

_load_damping_param = {
    "d3bj":   libdftd3.dftd3_load_rational_damping,      #RationalDampingParam,
    "d3zero": libdftd3.dftd3_load_zero_damping,          #ZeroDampingParam,
    "d3bjm":  libdftd3.dftd3_load_mrational_damping,     #ModifiedRationalDampingParam,
    "d3mbj":  libdftd3.dftd3_load_mrational_damping,     #ModifiedRationalDampingParam,
    "d3zerom":libdftd3.dftd3_load_mzero_damping,         #ModifiedZeroDampingParam,
    "d3mzero":libdftd3.dftd3_load_mzero_damping,         #ModifiedZeroDampingParam,
    "d3cso":  libdftd3.dftd3_load_cso_damping,           #CSODampingParam,
    "d3op":   libdftd3.dftd3_load_optimizedpower_damping #OptimizedPowerDampingParam,
}

class _d3_restype(ctypes.Structure):
    pass

_d3_p = ctypes.POINTER(_d3_restype)

libdftd3.dftd3_new_error.restype                   = _d3_p
libdftd3.dftd3_new_structure.restype               = _d3_p
libdftd3.dftd3_load_optimizedpower_damping.restype = _d3_p
libdftd3.dftd3_load_mzero_damping.restype          = _d3_p
libdftd3.dftd3_load_mrational_damping.restype      = _d3_p
libdftd3.dftd3_load_zero_damping.restype           = _d3_p
libdftd3.dftd3_load_rational_damping.restype       = _d3_p
libdftd3.dftd3_load_cso_damping.restype            = _d3_p
libdftd3.dftd3_new_d3_model.restype                = _d3_p
libdftd3.dftd3_check_error.restype                 = ctypes.c_int

def error_check(err):
    if libdftd3.dftd3_check_error(err):
        size = ctypes.c_int(2048)
        message = ctypes.create_string_buffer(2048)
        libdftd3.dftd3_get_error(err, message, ctypes.byref(size))
        raise RuntimeError(message.value.decode())

class DFTD3Dispersion(lib.StreamObject):
    def __init__(self, mol, xc, version='d3bj', atm=False):
        xc_lc = xc.lower().replace('-', '').replace('_', '').encode()
        self._disp = None
        self._mol = None
        self._param = None

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

            self._disp = libdftd3.dftd3_new_d3_model(err, self._mol)
            error_check(err)

            self._param = _load_damping_param[version](
                err,
                xc_lc,
                ctypes.c_bool(atm))
            error_check(err)
        finally:
            libdftd3.dftd3_delete_error(ctypes.byref(err))
        
    def __del__(self):
        if self._param:
            libdftd3.dftd3_delete_param(ctypes.byref(self._param))
        if self._disp:
            libdftd3.dftd3_delete_model(ctypes.byref(self._disp))
        if self._mol:
            libdftd3.dftd3_delete_structure(ctypes.byref(self._mol))

    def set_realspace_cutoff(self, disp2, disp3, cn, width2=0.0, width3=0.0):
        '''Set realspace cutoffs and smoothing widths in Bohr.

        Args:
            disp2 : float
                Cutoff radius for two-body dispersion interactions.
            disp3 : float
                Cutoff radius for three-body Axilrod-Teller-Muto interactions.
            cn : float
                Cutoff radius for evaluating coordination numbers.
            width2, width3 : float, optional
                Smoothing widths for the two-body and three-body cutoffs,
                respectively. For 0 < width < cutoff, the switching function
                decreases from one to zero between cutoff-width and cutoff.
                Otherwise smoothing is disabled. The default is 0.0.
        '''
        err = libdftd3.dftd3_new_error()
        try:
            libdftd3.dftd3_set_model_realspace_cutoff_smooth(
                err,
                self._disp,
                ctypes.c_double(disp2),
                ctypes.c_double(disp3),
                ctypes.c_double(cn),
                ctypes.c_double(width2),
                ctypes.c_double(width3))
            error_check(err)
        finally:
            libdftd3.dftd3_delete_error(ctypes.byref(err))

    def set_ewald_summation(self, rank=0, tolerance=0.0, kcut=0.0, mesh=0):
        '''Enable Ewald summation for 3D PBC with BJ or zero damping.

        The two-body realspace cutoff is ignored once Ewald summation is
        enabled. Analytical Hessians are not supported with Ewald summation.

        Args:
            rank : int, optional
                Rank of the reference C6 coefficient expansion. Non-positive
                values select the rank from tolerance. The default is 0.
            tolerance : float, optional
                Maximum relative error of the reference C6 coefficients when
                rank is selected automatically. Non-positive values use the
                upstream default of 1e-4. The default is 0.0.
            kcut : float, optional
                Reciprocal-space cutoff in Bohr**-1. Non-positive values select
                the cutoff from the damping radii. The default is 0.0.
            mesh : int, optional
                Number of mesh points per direction, rounded up to a power of
                two. Zero selects a mesh from kcut; negative values use a
                direct reciprocal-lattice sum. The default is 0.
        '''
        err = libdftd3.dftd3_new_error()
        try:
            libdftd3.dftd3_set_model_ewald(
                err,
                self._disp,
                ctypes.c_int(rank),
                ctypes.c_double(tolerance),
                ctypes.c_double(kcut),
                ctypes.c_int(mesh))
            error_check(err)
        finally:
            libdftd3.dftd3_delete_error(ctypes.byref(err))

    def get_hessian(self):
        '''Return D3 dispersion energy and analytical Cartesian Hessian.

        Returns:
            A dict with keys 'energy' (scalar array in Hartree) and 'hessian'
            ((3N,3N) array in Hartree/Bohr**2), where N is the number of atoms.
            Both Hessian axes follow the order x1, y1, z1, x2, y2, z2, ...

        Raises:
            RuntimeError if Ewald summation has been enabled.
        '''
        _energy = np.array(0.0, dtype=np.double)
        _hessian = np.zeros((3*self.natm,3*self.natm))
        err = libdftd3.dftd3_new_error()
        try:
            libdftd3.dftd3_get_dispersion_hessian(
                err,
                self._mol,
                self._disp,
                self._param,
                _energy.ctypes.data_as(ctypes.c_void_p),
                _hessian.ctypes.data_as(ctypes.c_void_p))
            error_check(err)
        finally:
            libdftd3.dftd3_delete_error(ctypes.byref(err))
        return dict(energy=_energy, hessian=_hessian)

    def get_dispersion(self, grad=False):
        res = {}
        _energy = np.array(0.0, dtype=np.double)
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
            libdftd3.dftd3_get_dispersion(
                err,
                self._mol,
                self._disp,
                self._param,
                _energy.ctypes.data_as(ctypes.c_void_p),
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
