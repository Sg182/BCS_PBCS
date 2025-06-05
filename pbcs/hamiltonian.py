from numpy import empty, zeros, ones, eye, diag
from pyscf import gto, scf, ao2mo


class Hubbard:
    """
    Hamiltonian for the two-body pairing channel
    """

    def __init__(self, nmo, nelec, t, U, periodic=True, bias=None):

        self.nmo = nmo
        self.nelec = nelec
        self.t = t
        self.U = U
        self.periodic = periodic

        mol = gto.M(verbose=4)
        mol.nelectron = nelec
        mol.incore_anyway = True

        mf = scf.RHF(mol)
        hcore = self.build_hcore()
        if bias is not None:
            hcore += diag(bias)
        # print('hcore')
        # print(hcore)
        mf.get_hcore = lambda *args: hcore
        mf.get_ovlp = lambda *args: eye(nmo)
        mf._eri = ao2mo.restore(8, self.build_eri(), nmo)
        mf.kernel()

        self.hf_mo_energy = mf.mo_energy

        eri_mo = ao2mo.full(
            mf._eri, mf.mo_coeff, compact=False
        ).reshape(nmo, nmo, nmo, nmo)
        self.eri_mulliken = eri_mo

        cmat = mf.mo_coeff
        self.h = cmat.T @ hcore @ cmat
        self.h_diag = diag(self.h)

        v = empty((nmo, nmo))
        for p in range(nmo):
            for q in range(nmo):
                v[p, q] = eri_mo[p, q, p, q]
        self.v = v

        w = empty((nmo, nmo))
        for p in range(nmo):
            for q in range(nmo):
                w[p, q] = 2.0 * eri_mo[p, p, q, q] - eri_mo[p, q, q, p]
        self.w = w

        self.mol = mol
        self.rhf = mf

    def build_hcore(self):

        n = self.nmo
        t = self.t

        hcore = zeros((n, n))
        for i in range(n - 1):
            hcore[i, i + 1] = hcore[i + 1, i] = -t
        if self.periodic:
            hcore[n - 1, 0] = hcore[0, n - 1] = -t

        return hcore

    def build_eri(self):

        n = self.nmo
        U = self.U

        eri = zeros((n, n, n, n))
        for i in range(n):
            eri[i, i, i, i] = U

        return eri


class Pairing:
    """
    Reduced BCS (or pairing) Hamiltonian
    """

    def __init__(self, nmo, nelec, one_body, G):

        self.nmo = nmo
        assert nelec % 2 == 0
        self.nelec = nelec

        self.h_diag = one_body
        self.G = G

        self.v = - G * ones((nmo, nmo))
        self.w = zeros((nmo, nmo))


class XXZ:
    """
    Heisenberg XXZ model

    Notice:

    In the spin-to-pairing mapping, Sz_p Sz_q = (N_p N_q - N_p - N_q + 1)/4
    The constant term 1/4 is not included in the spin Hamiltonian implemented
    below and should be added to the final energy after the AGP iterations.
    """

    def __init__(self, nmo, delta, periodic=False):

        self.nmo = nmo
        self.nelec = nmo

        h_diag = -0.5 * ones((nmo))

        if not periodic:
            h_diag[0] = -0.25
            h_diag[-1] = -0.25

        self.h_diag = delta * h_diag

        vmat = zeros((nmo, nmo))
        for i in range(nmo - 1):
            j = i + 1
            vmat[i, j] = 0.5
            vmat[j, i] = 0.5
        if periodic:
            vmat[0, nmo - 1] = vmat[nmo - 1, 0] = 0.5

        self.v = vmat
        self.w = delta * vmat


class J1J2XXZ:
    """
    Anisotropic J1-J2 model
    """

    def __init__(self, nmo, delta, j2, periodic=False):

        self.nmo = nmo
        self.nelec = nmo

        h_diag = -(0.5 + 0.5 * j2) * ones((nmo))
        if not periodic:
            h_diag[0] = -0.25 * (1 + j2)
            h_diag[-1] = -0.25 * (1 + j2)
            h_diag[1] = -0.5 - 0.25 * j2
            h_diag[-2] = -0.5 - 0.25 * j2
        self.h_diag = delta * h_diag

        vmat = zeros((nmo, nmo))
        for i in range(nmo - 1):
            j = i + 1
            vmat[i, j] = 0.5
            vmat[j, i] = 0.5
        for i in range(nmo - 2):
            j = i + 2
            vmat[i, j] = 0.5 * j2
            vmat[j, i] = 0.5 * j2
        if periodic:
            vmat[0, nmo - 1] = vmat[nmo - 1, 0] = 0.5
            vmat[0, nmo - 2] = vmat[nmo - 2, 0] = vmat[1, nmo - 1] \
                = vmat[nmo-1, 1] = 0.5 * j2

        self.v = vmat
        self.w = delta * vmat


def _squarefindneighbour(nx, ny, ix, iy, nn):
    """ Find nearest/next nearest neighbour for square lattice."""
    # Nearest neighbour
    if nn == 1:
        neighbour = []
        for jx in [ix + 1, ix - 1]:
            jx = jx % nx
            jy = iy
            j = jx*ny+jy
            neighbour.append(j)
        for jy in [iy + 1, iy - 1]:
            jy = jy % ny
            jx = ix
            j = jx * ny + jy
            neighbour.append(j)
        return neighbour

    # Next nearest neighbour
    if nn == 2:
        neighbour = []
        for jx in [ix + 1, ix - 1]:
            for jy in [iy + 1, iy - 1]:
                jx = jx % nx
                jy = jy % ny
                j = jx * ny + jy
                neighbour.append(j)
        return neighbour


class J1J2Square:
    """
    Square J1-J2 model
    """

    def __init__(self, nx, ny, j2, periodic=True):
        nmo = nx * ny
        self.nmo = nmo
        self.nelec = nmo
        if not periodic:
            print('J1J2 Square with OBC is not implemented yet.')
            exit()
        h_diag = -(0.5 + 0.5 * j2) * ones((nmo)) * 2.0

        self.h_diag = h_diag

        vmat = zeros((nmo, nmo))
        for ix in range(nx):
            for iy in range(ny):
                i = ix*ny + iy
                # nearest neibour
                for j in _squarefindneighbour(nx, ny, ix, iy, 1):
                    vmat[i, j] = 0.5
                # next nearest neibour
                for j in _squarefindneighbour(nx, ny, ix, iy, 2):
                    vmat[i, j] = 0.5 * j2

        self.v = vmat
        self.w = vmat


class GeneralSeniorityZero:
    """
    General Seniority-Zero Hamiltonian
    """

    def __init__(self, nmo, nelec, one_body, v, w):

        self.nmo = nmo
        assert nelec % 2 == 0
        self.nelec = nelec

        self.h_diag = one_body

        self.v = v
        self.w = w
