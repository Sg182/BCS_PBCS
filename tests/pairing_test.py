from pbcs import Pairing, PBCS, PairingChannel
from numpy import arange, diag, ones, zeros, sqrt, allclose
from numpy.linalg import eigh


def fix_eta(eta0, npair, maxit=32, tol=1e-5):
    '''Fix eta so that the average number of pairs is correct.

    This function was adapted from Tom's Fortran code.
    '''
    fac = 1.0
    x = 1.0 / sqrt(eta0 * eta0 + 1.0)
    y = x * eta0
    npair_err = y @ y - npair

    converged = False
    for _ in range(maxit):
        deriv = 2.0 / fac * (x * x) @ (y * y)
        fac -= npair_err / deriv
        x = 1.0 / sqrt(fac * fac * eta0 * eta0 + 1.0)
        y = fac * x * eta0
        npair_err = y @ y - npair
        if abs(npair_err) < tol:
            converged = True
            break

    eta = fac * eta0

    return eta, converged


def test_pairing():

    nmo = 12
    nelec = nmo
    G = 10.
    one_body = arange(nmo) + 1.0
    hamilt_pairing = Pairing(nmo, nelec, one_body, G)

    pairing = PairingChannel(hamilt_pairing)
    x1, y1, info1 = pairing.run(init_guess='extreme_agp', maxrun=16,
                                is_from_complex=False)
    x2, y2, info2 = pairing.run(init_guess='extreme_agp', maxrun=16,
                                is_from_complex=True)

    energy = -343.29882849

    assert abs(info1['obj_val'] - energy) < 1e-6
    assert abs(info2['obj_val'] - energy) < 1e-6
    assert abs(info1['g'] - nelec) < 1e-6
    assert abs(info2['g'] - nelec) < 1e-6


def test_pairing_dimer():

    nmo = 2
    nelec = nmo
    G = 4.
    one_body = arange(nmo) + 1.0

    dimer_doci = 2.0 * diag(one_body) - G * ones((nmo, nmo))
    evals, evecs = eigh(dimer_doci)

    energy_doci = evals[0]

    eta0, converged = fix_eta(evecs[:, 0], nelec//2)
    x0 = 1.0 / sqrt(1.0 + eta0 * eta0)
    y0 = x0 * eta0

    hamilt_pairing = Pairing(nmo, nelec, one_body, G)

    pbcs = PBCS(hamilt_pairing, x=x0, y=y0, is_complex=False)
    energy = pbcs.objective(zeros(nmo))
    grad = pbcs.gradient(zeros(nmo))

    assert abs(energy - energy_doci) < 1e-6
    assert allclose(grad, 0.0)

    pairing = PairingChannel(hamilt_pairing)
    x, y, info = pairing.run(init_guess='extreme_agp', maxrun=16,
                             is_from_complex=False)
    eta = y / x

    assert abs(info['obj_val'] - energy_doci) < 1e-6
    assert abs(info['g'] - nelec) < 1e-6
    assert allclose(eta, eta0) or allclose(eta, -eta0)


def test_pairing_bcs():

    nmo = 12
    nelec = nmo
    G = 0.5
    one_body = arange(nmo) + 1.0
    hamilt_pairing = Pairing(nmo, nelec, one_body, G)

    pairing = PairingChannel(hamilt_pairing)

    x, y, info = pairing.run(init_guess='perturbed', is_from_complex=False,
                             bcs_only=True)

    energy = 38.2377422

    assert abs(info['obj_val'] - energy) < 1e-6
    assert abs(info['g'] - nelec) < 1e-6


