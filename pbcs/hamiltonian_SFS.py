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

    def __init__(self, nmo, nelec, one_body, G,Lambda):

        self.nmo = nmo
        assert nelec % 2 == 0
        self.nelec = nelec
        self.Lambda = Lambda

        self.h_diag = one_body - self.Lambda
        self.G = G

        self.v = -G * ones((nmo, nmo))
        self.w = zeros((nmo, nmo))


class XXZ:
    """
    Heisenberg XXZ model

    Notice:

    In the spin-to-pairing mapping, Sz_p Sz_q = (N_p N_q - N_p - N_q + 1)/4
    The constant term 1/4 is not included in the spin Hamiltonian implemented
    below and should be added to the final energy after the AGP iterations.
    """

    def __init__(self, nmo, delta, periodic=True):

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
            vmat[i, j] += 0.5
            vmat[j, i] += 0.5
        if periodic:
            vmat[0,nmo-1] +=0.5
            vmat[nmo-1,0] +=0.5

        self.v = vmat
        self.w = ( delta * vmat)

        self.Enuc = delta/4 * nmo
        if (periodic == False):
             self.Enuc = self.Enuc - delta/4
        print("H010",self.h_diag)
        print("H101",self.v)
        print("H020",self.w)

class J1J2XXZ:
    """
    Anisotropic J1-J2 model
    """

    def __init__(self, nmo, delta, j2, periodic=True):

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
            vmat[0, nmo - 2] = vmat[nmo - 2 ,0] = vmat[1, nmo -1] \
             = vmat[nmo-1, 1] = 0.5 * j2

        self.v = vmat
        self.w = delta * vmat
        self.Enuc = delta/4 * nmo +  j2*(1/4)*2*(1/2)*nmo
        if (periodic == False):
             self.Enuc = self.Enuc - delta/4



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

        if (periodic==True):
            self.Enuc = nmo*(1/2) + j2*(nmo*(1/2))


        self.v = vmat
        self.w = vmat

class XXZSquare:
    

    def __init__(self, nx, ny,delta, periodic=True):
        
        nmo = nx * ny
        self.nmo = nmo
        self.nelec = nmo
        if not periodic:
            print('J1J2 Square with OBC is not implemented yet.')
            exit()
        h_diag = -(0.5) * ones((nmo)) * 2.0

        self.h_diag = h_diag*delta

        vmat = zeros((nmo, nmo))
        for ix in range(nx):
            for iy in range(ny):
                i = ix*ny + iy
                # nearest neibour
                for j in _squarefindneighbour(nx, ny, ix, iy, 1):
                    vmat[i, j] = 0.5
                

        self.v = vmat
        self.w = vmat*delta
        self.Enuc = delta*2 * nmo/4              ## write the Enunc for 2D
        if (periodic == False):
             self.Enuc = self.Enuc



def _triangular_findneighbour(nx, ny, ix, iy):
    """Find nearest neighbours for a triangular lattice."""
    neighbour = []

    # Determine the neighbors based on even or odd row
    if iy % 2 == 0:  # even row
        potential_neighbours = [
            (ix + 1, iy),     # right
            (ix, iy + 1),     # top-right
            (ix - 1, iy + 1), # top-left
            (ix - 1, iy),     # left
            (ix - 1, iy - 1), # bottom-left
            (ix, iy - 1)      # bottom-right
        ]
    else:  # odd row
        potential_neighbours = [
            (ix + 1, iy),     # right
            (ix + 1, iy + 1), # top-right
            (ix, iy + 1),     # top-left
            (ix - 1, iy),     # left
            (ix, iy - 1),     # bottom-left
            (ix + 1, iy - 1)  # bottom-right
        ]

    for jx, jy in potential_neighbours:
        # Apply periodic boundary conditions
        jx = jx % nx
        jy = jy % ny
        # Calculate the index
        j = jx * ny + jy
        neighbour.append(j)

    return neighbour


def _honeycomb_findneighbour(nx, ny, ix, iy):
    """Find nearest neighbors for a honeycomb lattice."""
    neighbour = []

    # Determine if the point is on sub-lattice A or B
    if (ix + iy) % 2 == 0:  # sub-lattice A
        potential_neighbours = [
            (ix + 1, iy),       # right
            (ix, iy + 1),       # up-right
            (ix - 1, iy + 1)    # up-left
        ]
    else:  # sub-lattice B
        potential_neighbours = [
            (ix - 1, iy),       # left
            (ix, iy - 1),       # down-left
            (ix + 1, iy - 1)    # down-right
        ]

    for jx, jy in potential_neighbours:
        # Apply periodic boundary conditions
        jx = jx % nx
        jy = jy % ny
        # Calculate the index
        j = jx * ny + jy
        neighbour.append(j)

    return neighbour

class XXZtriangular:
    """
    Square J1-J2 model
    """

    def __init__(self,delta, nx, ny, periodic=True):
        nx = int(nx)
        ny = int(ny)
        nmo = nx * ny
        self.nmo = nmo
        self.nelec = nmo
        if not periodic:
            print('J1J2 Square with OBC is not implemented yet.')
            exit()
        h_diag = -(0.5) * ones((nmo)) * 3.0

        self.h_diag = h_diag*delta

        vmat = zeros((nmo, nmo))
        for ix in range(nx):
            for iy in range(ny):
                i = ix*ny + iy
                # nearest neibour
                for j in _triangular_findneighbour(nx, ny, ix, iy):
                    vmat[i, j] = 0.5


        self.v = vmat
        self.w = vmat*delta
        
        self.Enuc = delta*(6/8) * nmo              ## write the Enunc for 2D
        if (periodic == False):
             self.Enuc = self.Enuc

class XXZhoneycomb:
    """
    Square J1-J2 model
    """

    def __init__(self,delta, nx, ny, periodic=True):
        nx = int(nx)
        ny = int(ny)
        nmo = nx * ny
        self.nmo = nmo
        self.nelec = nmo
        if not periodic:
            print('J1J2 Square with OBC is not implemented yet.')
            exit()
        h_diag = -(0.5) * ones((nmo)) * 1.5

        self.h_diag = h_diag*delta

        vmat = zeros((nmo, nmo))
        for ix in range(nx):
            for iy in range(ny):
                i = ix*ny + iy
                # nearest neibour
                for j in _honeycomb_findneighbour(nx, ny, ix, iy):
                    vmat[i, j] = 0.5


        self.v = vmat
        self.w = vmat*delta

class SFS_1D_XXZ:
    def __init__(self, nmo, delta,periodic=True):
        self.nmo = nmo
        self.delta = delta

        # Now build the integrals h100, h001 and so on

        h100 = -0.25 * np.ones(nmo)
        h001 = -0.25*np.ones(nmo)

        #build h010
        
        if periodic:
            h010 = -0.5*delta*np.ones(nmo)
        
        else:
            h010 = -0.5*delta*np.ones(nmo)
            h010[0] = -0.25*delta
            h010[-1] = -0.25*delta

        #build h110

        h110 = np.zeros((nmo,nmo))

        if periodic:
            p = np.arange(nmo)
            h110[p,(p+1)%nmo] += 0.5
            h110[p,(p-1)%nmo] += 0.5
        else:
            p = np.arange(1,nmo-1)
            h110[p,p+1] += 0.5
            h110[p,p-1] += 0.5 

        
        #build h011

        h011 = h110

        #build h120

        h120 = np.zeros((nmo,nmo,nmo))

        if periodic:
            
            for p in range(nmo):

                q = (p-1) % nmo
                r = (p+1) % nmo
                h120[p,q,r] += -0.25
                h120[p,r,q] += -0.25

        else:

            for p in range(1, nmo-1):
                q = p-1
                r = p+1
                h120[p,q,r] += -0.25
                h120[p,r,q] += -0.25

       #build h021

        h021 = np.zeros((nmo,nmo,nmo))

        if periodic:
            for p in range(nmo):
                r = (p-1) % nmo
                q = (p+1) % nmo
                h021[r,q,p] += -0.25
                h021[q,r,p] += -0.25

        else:
            for p in range(1, nmo-1):
                r = p-1
                q = p+1
                h021[r,q,p] += -0.25
                h021[q,r,p] += -0.25



        #build h020

        h020 = np.zeros((nmo,nmo))
    
        if periodic:
            p = np.arange(nmo)
            q = (p+2)%nmo
            h020[p,q] += 0.125*delta
            h020[q,p] += 0.125*delta
        else:
            if nmo >= 3:
                p = np.arange(nmo-2)                           #I am dividing by 1/2 to symmetrize h[p,q]
                q = p+2
                h020[p,q] += 0.125*delta           
                h020[q,p] += 0.125*delta
      
      #build the scalar h000
        if periodic:
            h000 = delta*0.25*nmo
        else:
            h000 = delta*0.25*(nmo-1)
        





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
        self.w = e
