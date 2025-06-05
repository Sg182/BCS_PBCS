Usage
=====
The main class is `PairingChannel`, which is a driver of the constrained optimizer.
It needs to be initialized by an instance of a seniority-conserving Hamiltonian,
as implemented the in the `hamiltonian.py` file. The `Hubbard` class is the
Hubbard model Hamiltonian in RHF basis, while the `Pairing` class is the pairing
or reduced BCS Hamiltonian.  The users may implement their own seniority-conserving
Hamiltonian in the form of

$$H = \sum_p h_p N_p + \sum_{pq} v_{pq} P^\dagger_p P_q + \frac{1}{4} \sum_{p\neq q} w_{pq} N_p N_q$$

where *v* is Hermitian and *w* real symmetric. The code can be readily extended to
higher-body Hamiltonians if needed.

After initializing `PairingChannel`, just use the `run` method to perform the
*successive Thouless rotations*. Internally, an instance of the `PBCS` class inplemeted
in `pbcs_ipopt.py` is created. The `PBCS` class contains the implementations of the
PBCS energy and gradient as well as the number expectation value constraint.


Installation
============
The code uses the python wrapper of the IPOPT optimizer, `cyipopt`, which is
[available on GitHub](https://github.com/matthias-k/cyipopt).
After installing `cyipopt`, run the following command in the current directory:
```python
pip install .
```
