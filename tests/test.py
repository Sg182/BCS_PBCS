# %%
from pbcs import Hubbard, Pairing, XXZ, PBCS, PairingChannel
from numpy import arange, diag, ones, zeros, sqrt, exp
from numpy.linalg import eigh
from numpy.random import rand

# nmo = 16
# nelec = 18
# U = 4.0
# hamilt = Hubbard(nmo, nelec, 1.0, U, periodic=False)

# # %%
# pairing = PairingChannel(hamilt)
# x, y, info = pairing.run(is_from_complex=True)

# # %%
# nmo = 12
# nelec = nmo
# G = 10.
# one_body = arange(nmo) + 1.0
# hamilt_pairing = Pairing(nmo, nelec, one_body, G)

# # %%
# pairing = PairingChannel(hamilt_pairing)
# x, y, info = pairing.run(init_guess='extreme_agp', is_from_complex=True)
# print(y / x)

# # %%
# nmo = 2
# nelec = nmo
# G = 5.
# one_body = arange(nmo) + 1.0
# hamilt_pairing = Pairing(nmo, nelec, one_body, G)

# # %%
# pairing = PairingChannel(hamilt_pairing)
# x, y, info = pairing.run(init_guess='extreme_agp', is_from_complex=False)
# print(y / x)

# # %%
# dimer_doci = 2.0 * diag(one_body) - G * ones((nmo, nmo))
# evals, evecs = eigh(dimer_doci)
# print(evals)
# print(evecs)

# # %%
# # Adapted from Tom's Fortran code
# def fix_eta(eta0, npair, maxit=32, tol=1e-5):

#     fac = 1.0
#     x = 1.0 / sqrt(eta0 * eta0 + 1.0)
#     y = x * eta0
#     npair_err = y @ y - npair

#     converged = False
#     for _ in range(maxit):
#         deriv = 2.0 / fac * (x * x) @ (y * y)
#         fac -= npair_err / deriv
#         x = 1.0 / sqrt(fac * fac * eta0 * eta0 + 1.0)
#         y = fac * x * eta0
#         npair_err = y @ y - npair
#         if abs(npair_err) < tol:
#             converged = True
#             break

#     eta = fac * eta0
#     return eta, converged

# # %%
# eta0, converged = fix_eta(evecs[:, 0], nelec//2)
# x0 = 1.0 / sqrt(1.0 + eta0 * eta0)
# y0 = x0 * eta0

# # %%
# pbcs = PBCS(hamilt_pairing, x=x0, y=y0, is_complex=False)
# en = pbcs.objective(zeros(nmo))
# grad = pbcs.gradient(zeros(nmo))
# print(en , grad)

# # %%


# %%
nmo = 14
# nmo = 16
hamilt = XXZ(nmo, 6.0)

# %%
pairing = PairingChannel(hamilt)
x0 = sqrt(0.5) * ones(nmo, dtype=complex)
y0 = x0.copy()
y0 *= exp(0.1j * rand(nmo))
for i in range(0, nmo, 2):
    y0[i] *= -1.0

# x, y, info = pairing.run(x0=x0, y0=y0, is_from_complex=True)
x, y, info = pairing.run(x0=x0, y0=y0, is_from_complex=True,
                         maxrun=50, tol=1e-9)

# %%
assert info["status"] == 0
print("Optimized eta")
print(y / x)
# %%
