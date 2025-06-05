from numpy import exp, sum

MAXIT = 128
TOL = 1e-10
KB = 3.16681536e-6


def fermi(npair, eig, temp):
    """
    Optimize the Fermi energy and compute orbital occupations
    according to Fermi-Dirac distribution.

    This function was translated from Carlos's Matlab code.

    Parameters
    ----------

    npair
        Number of pairs of electrons

    eig
        Array of orbital energies

    temp
        Temperature used for thermal broadening

    Returns
    -------

    occ
        Array of occupations
    ef
        Fermi energy
    """
    beta = 1.0 / (KB * temp)

    xl = eig[npair - 1]
    xr = eig[npair]

    converged = False
    for _ in range(MAXIT):

        xm = 0.5 * (xr + xl)

        summ = sum(1.0 / (1.0 + exp(beta * (eig - xm)))) - npair
        suml = sum(1.0 / (1.0 + exp(beta * (eig - xl)))) - npair
        sumr = sum(1.0 / (1.0 + exp(beta * (eig - xr)))) - npair

        if suml * sumr > 0.0:
            # Expand interval
            fac = xr - xm
            xl = xm - 2.0 * fac
            xr = xm + 2.0 * fac
        else:
            # Bisection
            if abs(summ) < TOL:
                converged = True
                break
            else:
                if (suml * summ < 0.0):
                    xr = xm
                else:
                    xl = xm

    if converged:
        ef = xm
    else:
        ef = 0.5 * (eig[npair - 1] + eig[npair])
        raise UserWarning('Failed to determine Fermi energy.')

    # Build the vector of occupations using the computed Fermi energy
    occ = 1.0 / (1.0 + exp(beta * (eig - ef)))

    return occ, ef
