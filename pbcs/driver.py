from numpy import finfo, sqrt, ones, zeros, argmax, abs
from numpy.linalg import norm
from cyipopt import Problem
from .pbcs_ipopt import PBCS


MAX = finfo(float).max


class PairingChannel:

    def __init__(self, hamilt, ngrid=32, maxrun=8, maxiter=32):

        self.maxrun = maxrun
        self.maxiter = maxiter

        self.nmo = hamilt.nmo
        self.nelec = hamilt.nelec
        self.ngrid = ngrid
        self.hamilt = hamilt

        self.eta = None
        self.pbcs = None

    def _optimize(self, maxiter=None, tol=1e-7):

        nelec = self.nelec
        nmo = self.nmo
        if self.pbcs is None:
            raise AttributeError('pbcs not initialized')
        else:
            pbcs = self.pbcs

        if pbcs.is_complex:
            n = nmo * 2
        else:
            n = nmo

        eta_vec0 = zeros(n)
        lb = - MAX * ones(n)
        ub = MAX * ones(n)

        cl = [nelec]
        cu = [nelec]

        nlp = Problem(n=n, m=1, problem_obj=pbcs, lb=lb, ub=ub, cl=cl, cu=cu)
        # nlp.add_option('derivative_test', 'first-order')
        # nlp.add_option('derivative_test', 'only-second-order')
        # nlp.add_option('derivative_test', 'second-order')
        nlp.add_option('mu_strategy', 'adaptive')
        nlp.add_option('tol', tol)
        if maxiter is None:
            maxiter = self.maxiter
        nlp.add_option('max_iter', maxiter)

        eta_vec, info = nlp.solve(eta_vec0)

        print("Solution of the primal variables: eta =\n%s\n" % repr(eta_vec))
        print(
            "Solution of the dual variables: lambda = %s\n"
            % repr(info['mult_g'])
        )
        print("Energy = %s\n" % repr(info['obj_val']))

        x, y, _ = pbcs.update_orb(eta_vec)

        # Global eta
        self.eta = (y / x).conj()

        return x, y, info

    def run(self, x0=None, y0=None, is_complex=False, is_from_complex=False,
            maxrun=None, maxiter=None, init_guess='thermal',
            bcs_only=False, tol=1e-7):

        pbcs = PBCS(self.hamilt, ngrid=self.ngrid, x=x0, y=y0,
                    init_guess=init_guess,
                    is_complex=(is_complex or is_from_complex),
                    bcs_only=bcs_only)
        self.pbcs = pbcs

        if maxrun is None:
            maxrun = self.maxrun

        if is_from_complex and not is_complex:
            for it_macro in range(maxrun):
                x, y, info = self._optimize(maxiter=maxiter, tol=tol)
                pbcs.x = x
                pbcs.y = y
                if info['status'] == 0:
                    break

            ncycles = it_macro + 1

            eta_complex = self.eta
            idx = argmax(abs(eta_complex))
            tmp = eta_complex[idx]
            phase_inv = tmp.conj() / abs(tmp)
            eta = phase_inv * eta_complex
            assert norm(eta.imag) < 1e-5

            eta = eta.real

            x = 1.0 / sqrt(1.0 + eta**2)
            y = eta * x

            pbcs.x = x
            pbcs.y = y
            pbcs.is_complex = False

        for it_macro in range(maxrun):
            x, y, info = self._optimize(maxiter=maxiter, tol=tol)
            pbcs.x = x
            pbcs.y = y
            if info['status'] == 0:
                if is_from_complex:
                    info['ncycles'] = ncycles
                else:
                    info['ncycles'] = it_macro + 1
                break

        return x, y, info
