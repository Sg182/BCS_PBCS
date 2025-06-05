
from pbcs import Hubbard, PairingChannel


def test_hubbard():

    nmo = 16
    nelec = 18
    U = 4.0
    hamilt = Hubbard(nmo, nelec, 1.0, U, periodic=False)

    pairing = PairingChannel(hamilt)
    x1, y1, info1 = pairing.run(maxrun=16)
    x2, y2, info2 = pairing.run(maxrun=16, is_from_complex=True)

    energy = 0.47436557

    assert abs(info1['obj_val'] - energy) < 1e-6
    assert abs(info2['obj_val'] - energy) < 1e-6
    assert abs(info1['g'] - nelec) < 1e-6
    assert abs(info2['g'] - nelec) < 1e-6

