from numpy.random import rand
from numpy import (
    array, arange, ones, empty, prod, sqrt, exp, real, concatenate, pi, copy,
    diag, block, tril_indices, outer, einsum, cos
)
from numpy.linalg import norm
from .fermi import fermi


class PBCS:

    def __init__(self, hamilt, x=None, y=None,
                 init_guess='thermal', hf_mo_energy=None, temp=1e5,
                 bcs_only=False, ngrid=32, is_complex=True, max_d_norm=1e3):

        self.h = hamilt.h_diag
        self.v = hamilt.v
        self.w = hamilt.w
        self.nelec = hamilt.nelec

        self.ngrid = ngrid
        self.bcs_only = bcs_only
        self.is_complex = is_complex
        self.max_d_norm = max_d_norm

        if x is None or y is None:
            if init_guess.lower() == 'thermal':
                if hf_mo_energy is None:
                    hf_mo_energy = hamilt.hf_mo_energy
                self.x, self.y = self.init_guess_thermal(hf_mo_energy, temp)
            elif init_guess.lower() == 'extreme_agp':
                self.x, self.y = self.init_guess_extreme_agp()
            elif init_guess.lower() == 'perturbed':
                self.x, self.y = self.init_guess_perturbed()
            else:
                raise ValueError('Inappropriate init_guess')
        else:
            self.x, self.y = x, y

    def init_guess_thermal(self, hf_mo_energy, temp):
        nmo = len(self.h)
        if hf_mo_energy is None:
            hf_mo_energy = arange(nmo, dtype=float)

        assert len(hf_mo_energy) == nmo

        occ, _ = fermi(self.nelec // 2, hf_mo_energy, temp)

        x = sqrt(occ)
        y = sqrt(1.0 - occ)
        if self.is_complex:
            x = x * exp(2.0j * pi * rand(nmo))
            y = y * exp(2.0j * pi * rand(nmo))

        return x, y

    def init_guess_extreme_agp(self):

        nmo = len(self.h)
        x = sqrt(0.5) * ones(nmo)
        y = copy(x)
        if self.is_complex:
            x = x * exp(2.0j * pi * rand(nmo))
            y = y * exp(2.0j * pi * rand(nmo))

        return x, y

    def init_guess_perturbed(self):
        nmo = len(self.h)
        y = empty(nmo)
        nocc = self.nelec // 2
        y[:nocc] = cos(0.1)
        y[nocc:] = sqrt((nocc - cos(0.1)**2 * nocc) / (nmo - nocc))
        x = sqrt(1.0 - y * y)

        return x, y

    def update_orb(self, eta_vec):
        '''Compute the x and y values after eta rotation.'''

        nmo = len(self.h)
        if self.is_complex:
            eta = eta_vec[:nmo] + eta_vec[nmo:] * 1.0j
        else:
            eta = eta_vec

        ll = sqrt(real(eta.conj() * eta) + 1.0)
        ll_inv = 1.0 / ll

        x = self.x
        y = self.y

        xx = (x - eta.conj() * y.conj()) * ll_inv
        yy = (y + eta.conj() * x.conj()) * ll_inv

        return xx, yy, ll

    def _setup_grids(self):
        '''Set up trapezoidal quadrature in (-pi, pi).'''
        if not self.bcs_only:
            ngrid = self.ngrid
            tmp = arange(float(ngrid))
            phis = 2.0 * pi * (tmp + 0.5) / ngrid - pi
            weights = 1.0 / ngrid * exp(-1.0j * self.nelec * phis)
        else:
            phis = [0.0]
            weights = [1.0]

        return phis, weights

    def objective(self, eta_vec):
        '''Compute energy of number-projected BCS.'''

        h = self.h
        v = self.v
        w = self.w

        xx, yy, _ = self.update_orb(eta_vec)

        phis, weights = self._setup_grids()

        en = 0.0
        denom = 0.0

        for phi, weight in zip(phis, weights):

            expiphi = exp(1.0j * phi)

            xx_phi = expiphi * xx
            yy_phi = expiphi.conj() * yy

            mu = xx * xx_phi.conj() + yy * yy_phi.conj()
            mu_inv = 1.0 / mu

            # 11 block of rho(phi)
            gamma = yy_phi.conj() * yy * mu_inv

            # 12 block of kappa(phi)
            chi = yy_phi.conj() * xx * mu_inv

            # 12 block of kappa_bar(phi).conj
            chi_bar_conj = xx_phi.conj() * yy * mu_inv

            # H(phi)
            en_phi = (2.0 * h + w @ gamma) @ gamma + chi @ v @ chi_bar_conj \
                + (gamma * gamma) @ (diag(v) - diag(w))

            # <Phi|Phi(phi)> overlap
            overlap = prod(expiphi * mu)

            tmp = weight * overlap
            denom += tmp.real
            en += (tmp * en_phi).real

        en /= denom

        return en

    def gradient(self, eta_vec):

        h = self.h
        v = self.v
        w = self.w

        xx, yy, ll = self.update_orb(eta_vec)

        phis, weights = self._setup_grids()

        en = 0.0
        denom = 0.0
        grad1 = 0.0j
        grad2 = 0.0j

        for phi, weight in zip(phis, weights):

            expiphi = exp(1.0j * phi)

            xx_phi = expiphi * xx
            yy_phi = expiphi.conj() * yy

            mu = xx * xx_phi.conj() + yy * yy_phi.conj()
            mu_inv = 1.0 / mu

            # 11 block of rho(phi)
            gamma = yy_phi.conj() * yy * mu_inv

            # 12 block of kappa(phi)
            chi = yy_phi.conj() * xx * mu_inv

            # 12 block of kappa_bar(phi).conj
            chi_bar_conj = xx_phi.conj() * yy * mu_inv

            # H(phi)
            en_phi = (2.0 * h + w @ gamma) @ gamma + chi @ v @ chi_bar_conj \
                + (gamma * gamma) @ (diag(v) - diag(w))

            # <Phi|Phi(phi)> overlap
            overlap = prod(expiphi * mu)

            tmp = weight * overlap
            denom += tmp.real
            en += (tmp * en_phi).real

            #
            # Gradient
            #
            mu_inv2 = mu_inv * mu_inv

            grad_mu = (expiphi - expiphi.conj()) * xx * yy
            grad_gamma = xx * yy * mu_inv2
            grad_chi = xx * xx * mu_inv2
            grad_chi_bar_conj = - yy * yy * mu_inv2
            grad_overlap = overlap * mu_inv * grad_mu

            grad_en_phi = 2.0 * (
                h + (diag(v) - diag(w)) * gamma + w @ gamma
            ) * grad_gamma + (
                chi_bar_conj @ v * grad_chi + v @ chi * grad_chi_bar_conj
            )

            grad1 += weight * (grad_overlap * en_phi +
                               overlap * grad_en_phi)
            grad2 += weight * grad_overlap

        en /= denom

        grad = (grad1 - en * grad2) / (ll * ll * denom)
        if self.is_complex:
            grad_vec = self.wirtinger_to_real_partial(grad)
        else:
            assert norm(grad.imag) < 1e-5
            grad_vec = 2.0 * grad.real

        return grad_vec

    @staticmethod
    def wirtinger_to_real_partial(dz):
        '''Convert the Wirtinger partial derivative `dz` to real partial
        derivatives `dxdy`.

        .. math::

            \frac{\partial}{\partial z} = 1/2
            (\frac{\partial}{\partial x} - i \frac{\partial}{\partial y})

        '''
        dxdy = 2.0 * concatenate((dz.real, -dz.imag))
        return dxdy

    def wirtinger_to_real_partial_hess(self, dzdz, dz_dz):

        nmo = len(dzdz)
        dxdx = 2.0 * (dz_dz.real + dzdz.real)
        if self.is_complex:
            dydy = 2.0 * (dz_dz.real - dzdz.real)
            dydx = 2.0 * (dz_dz.imag - dzdz.imag)

            res = block(
                [[dxdx, empty((nmo, nmo))],
                 [dydx, dydy]]
            )[tril_indices(2 * nmo)]
        else:
            res = dxdx[tril_indices(nmo)]

        return res

    def constraints(self, eta_vec):
        _, yy, _ = self.update_orb(eta_vec)
        nevl = 2.0 * real(yy.conj() @ yy)
        return array([nevl])

    def jacobian(self, eta_vec):
        xx, yy, ll = self.update_orb(eta_vec)
        grad_nevl = 2.0 * yy * xx / (ll * ll)
        if self.is_complex:
            grad_nevl_vec = self.wirtinger_to_real_partial(grad_nevl)
        else:
            assert norm(grad_nevl.imag) < 1e-5
            grad_nevl_vec = 2.0 * grad_nevl.real
        return grad_nevl_vec

    def hessianstructure(self):
        nmo = len(self.h)
        if self.is_complex:
            nmo *= 2
        return tril_indices(nmo)

    def hessian(self, eta_vec, lagrange, obj_factor):

        h = self.h
        v = self.v
        w = self.w

        nmo = len(h)
        if self.is_complex:
            eta = eta_vec[:nmo] + eta_vec[nmo:] * 1.0j
        else:
            eta = eta_vec

        x = self.x
        y = self.y
        xx, yy, ll = self.update_orb(eta_vec)

        phis, weights = self._setup_grids()

        en = 0.0
        denom = 0.0
        grad1 = 0.0j
        grad2 = 0.0j
        hess1 = 0.0j
        hess2 = 0.0j
        hess3 = 0.0j
        hess1_ = 0.0j
        hess2_ = 0.0j
        hess3_ = 0.0j

        for phi, weight in zip(phis, weights):

            expiphi = exp(1.0j * phi)

            xx_phi = expiphi * xx
            yy_phi = expiphi.conj() * yy

            mu = xx * xx_phi.conj() + yy * yy_phi.conj()
            mu_inv = 1.0 / mu

            # 11 block of rho(phi)
            gamma = yy_phi.conj() * yy * mu_inv

            # 12 block of kappa(phi)
            chi = yy_phi.conj() * xx * mu_inv

            # 12 block of kappa_bar(phi).conj
            chi_bar_conj = xx_phi.conj() * yy * mu_inv

            # H(phi)
            en_phi = (2.0 * h + w @ gamma) @ gamma + chi @ v @ chi_bar_conj \
                + (gamma * gamma) @ (diag(v) - diag(w))

            # <Phi|Phi(phi)> overlap
            overlap = prod(expiphi * mu)

            tmp = weight * overlap
            denom += tmp.real
            en += (tmp * en_phi).real

            #
            # Gradient
            #
            mu_inv2 = mu_inv * mu_inv

            grad_mu = (expiphi - expiphi.conj()) * xx * yy
            grad_gamma = xx * yy * mu_inv2
            grad_chi = xx * xx * mu_inv2
            grad_chi_bar_conj = - yy * yy * mu_inv2
            grad_overlap = overlap * mu_inv * grad_mu

            grad_en_phi = 2.0 * (
                h + (diag(v) - diag(w)) * gamma + w @ gamma
            ) * grad_gamma + (
                chi_bar_conj @ v * grad_chi + v @ chi * grad_chi_bar_conj
            )

            grad1 += weight * (grad_overlap * en_phi +
                               overlap * grad_en_phi)
            grad2 += weight * grad_overlap


            grad_mu_ = (expiphi - expiphi.conj()) * xx.conj() * yy.conj()
            grad_gamma_ = xx.conj() * yy.conj() * mu_inv2
            grad_chi_ = - expiphi * expiphi * yy.conj()**2 * mu_inv2
            grad_chi_bar_conj_ = expiphi.conj()**2 * xx.conj()**2 * mu_inv2
            grad_overlap_ = overlap * mu_inv * grad_mu_

            grad_en_phi_ = 2.0 * (
                h + (diag(v) - diag(w)) * gamma + w @ gamma
            ) * grad_gamma_ + (
                chi_bar_conj @ v * grad_chi_ + v @ chi * grad_chi_bar_conj_
            )

            #
            # Hessian
            #
            grad_mu_hat = (expiphi * yy * x - expiphi.conj() * xx * y) * ll
            mu_inv3 = mu_inv**3

            hess_gamma = -2.0 * xx * yy * grad_mu_hat * mu_inv3
            hess_chi = -2.0 * xx * xx * grad_mu_hat * mu_inv3
            hess_chi_bar_conj = 2.0 * yy * yy * grad_mu_hat * mu_inv3

            hess_gamma_ = (
                expiphi.conj() * xx * xx.conj() - expiphi * yy * yy.conj()
            ) * mu_inv3
            hess_chi_ = -2.0 * expiphi * xx * yy.conj() * mu_inv3
            hess_chi_bar_conj_ = -2.0 * (expiphi * xx).conj() * yy * mu_inv3

            hess_mu = -2.0 * (expiphi - expiphi.conj()) * xx * yy * eta.conj()
            tmp = mu_inv * grad_mu
            hess_overlap = overlap * (
                outer(tmp, tmp) + diag(-tmp * tmp + mu_inv * hess_mu)
            )

            hess_mu_ = (expiphi - expiphi.conj()) * (
                xx * xx.conj() - yy * yy.conj()
            )
            tmp_ = mu_inv * grad_mu_
            hess_overlap_ = overlap * (
                outer(tmp_, tmp) + diag(-tmp_ * tmp + mu_inv * hess_mu_)
            )

            tmp = einsum('p,q,pq->pq', grad_chi_bar_conj, grad_chi, v) \
                + einsum('p,q,pq->pq', grad_gamma, grad_gamma, w)
            hess_en_phi = diag(
                2.0 * (
                    h * hess_gamma
                    + (
                        gamma * hess_gamma + grad_gamma * grad_gamma
                    ) * (diag(v) - diag(w))
                    + gamma @ w * hess_gamma
                )
                + chi_bar_conj @ v * hess_chi
                + v @ chi * hess_chi_bar_conj
            ) + tmp + tmp.T

            tmp = outer(grad_overlap, grad_en_phi)
            hess1 += weight * (
                hess_overlap * en_phi + tmp + tmp.T + overlap * hess_en_phi
            )
            hess2 += weight * grad_overlap
            hess3 += weight * hess_overlap

            hess_en_phi_ = diag(
                2.0 * (
                    h * hess_gamma_
                    + (
                        gamma * hess_gamma_ + grad_gamma_ * grad_gamma
                    ) * (diag(v) - diag(w))
                    + gamma @ w * hess_gamma_
                )
                + chi_bar_conj @ v * hess_chi_
                + v @ chi * hess_chi_bar_conj_
            ) + einsum(
                'p,q,pq->pq', grad_chi_bar_conj_, grad_chi, v
            ) + einsum(
                'p,q,qp->pq', grad_chi_, grad_chi_bar_conj, v
            ) + 2.0 * einsum(
                'p,q,pq->pq', grad_gamma_, grad_gamma, w
            )

            hess1_ += weight * (
                hess_overlap_ * en_phi + outer(grad_en_phi_, grad_overlap)
                + outer(grad_overlap_, grad_en_phi) + overlap * hess_en_phi_
            )
            hess2_ += weight * grad_overlap_
            hess3_ += weight * hess_overlap_

        en /= denom
        grad = (grad1 - en * grad2) / denom

        tmp = outer(grad, hess2)
        hess = (hess1 - tmp - tmp.T - en * hess3) / denom
        hess_constr = -4.0 * diag(xx * yy * eta.conj())

        inv_ll2 = diag(1.0 / ll**2)
        hess = inv_ll2 @ (
            obj_factor * hess + lagrange[0] * hess_constr
        ) @ inv_ll2

        hess_ = (
            hess1_ - outer(grad.conj(), hess2) - outer(hess2_, grad)
            - en * hess3_
        ) / denom

        hess_constr_ = 2.0 * diag(xx * xx.conj() - yy * yy.conj())

        hess_ = inv_ll2 @ (
            obj_factor * hess_ + lagrange[0] * hess_constr_
        ) @ inv_ll2

        return self.wirtinger_to_real_partial_hess(hess, hess_)

    def intermediate(
        self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
        d_norm, regularization_size, alpha_du, alpha_pr, ls_trials
    ):
        # print('iter', iter_count, 'inf_pr', inf_pr, 'inf_du', inf_du)
        # print('d_norm', d_norm)

        return d_norm <= self.max_d_norm


