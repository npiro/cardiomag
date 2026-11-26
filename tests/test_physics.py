import numpy as np

from cardiomag.physics import leadfield as lf
from cardiomag.physics.sensors import SensorArray, ptb49_layout, kiel36_layout
from cardiomag.io.hearts import HalfSpaceSkin
from cardiomag.sim.eikonal import EdgeGraph, vm_from_at


def _cube_mesh(n=10, L=0.02):
    """Structured tetrahedral mesh of a cube (6 tets per cell), metres."""
    g = np.linspace(0, L, n + 1)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    idx = lambda i, j, k: (i * (n + 1) + j) * (n + 1) + k
    tets = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c = [idx(i + a, j + b, k + d) for a in (0, 1) for b in (0, 1) for d in (0, 1)]
                for t in ([0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]):
                    tets.append([c[q] for q in t])
    return pts, np.array(tets)


def test_leadfield_matches_direct_biot_savart():
    pts, tets = _cube_mesh(4)
    rng = np.random.default_rng(0)
    vm = rng.normal(size=pts.shape[0])
    fib = np.tile([1.0, 0, 0], (tets.shape[0], 1))
    sig = lf.conductivity_tensors(fib, np.zeros(tets.shape[0], int), {0: "X"}, {"X": (0.6, 0.3)}, k_cal=1.0)
    sensors = np.array([[0.01, 0.01, 0.08], [0.05, -0.02, 0.06]])
    normals = np.array([[0, 0, 1.0], [0.6, 0, 0.8]])
    W = lf.node_leadfield(pts, tets, sig, sensors, normals, chunk=37)
    cen, a = lf.element_dipole_operator(pts, tets, sig)
    m = np.einsum("eki,ek->ei", a, vm[tets])
    for s in range(2):
        R = sensors[s] - cen
        B = 1e-7 * (np.cross(m, R) / np.linalg.norm(R, axis=1)[:, None] ** 3).sum(0)
        assert np.isclose(W[s] @ vm, B @ normals[s], rtol=1e-9, atol=1e-30)


def test_uniform_gradient_gives_expected_dipole():
    pts, tets = _cube_mesh(3, L=0.01)
    vm = 2.0 * pts[:, 0]                    # dV/dx = 2 V/m
    fib = np.tile([1.0, 0, 0], (tets.shape[0], 1))
    sig = lf.conductivity_tensors(fib, np.zeros(tets.shape[0], int), {0: "X"}, {"X": (1.0, 1.0)}, k_cal=1.0)
    P = lf.dipole_operator_total(pts, tets, sig) @ vm
    expected = -lf.SIGMA_IL_PHYS * 2.0 * 0.01 ** 3          # -sigma * grad * volume
    assert np.allclose(P, [expected, 0, 0], rtol=1e-9, atol=1e-18)


def test_eikonal_isotropic_cube_speed():
    pts, tets = _cube_mesh(12, L=0.024)
    g = EdgeGraph(pts, tets, np.tile([1.0, 0, 0], (tets.shape[0], 1)))
    v = np.full(tets.shape[0], 0.5)                           # mm/ms
    at = g.solve(v, v, [(np.array([0]), 0.0)], factor=1.0)
    d = np.linalg.norm(pts - pts[0], axis=1) * 1e3
    far = d > 10
    ratio = at[far] / (d[far] / 0.5)
    assert 1.0 <= ratio.min() and ratio.max() < 1.25           # graph paths overestimate, bounded


def test_eikonal_anisotropy_and_block():
    pts, tets = _cube_mesh(10, L=0.02)
    g = EdgeGraph(pts, tets, np.tile([1.0, 0, 0], (tets.shape[0], 1)))
    vl, vt = np.full(tets.shape[0], 1.0), np.full(tets.shape[0], 0.25)
    at = g.solve(vl, vt, [(np.array([0]), 0.0)], factor=1.0)
    along = np.argmin(np.linalg.norm(pts - [0.02, 0, 0], axis=1))
    across = np.argmin(np.linalg.norm(pts - [0, 0.02, 0], axis=1))
    assert at[across] > 3 * at[along]
    blocked = pts[:, 0] > 0.015
    at_b = g.solve(vl, vt, [(np.array([0]), 0.0)], blocked_nodes=blocked, factor=1.0)
    assert np.isinf(at_b[blocked]).all()


def test_vm_template():
    at = np.array([10.0, np.inf])
    v = vm_from_at(at, np.array([0.0, 10.0, 30.0]))
    assert v[0, 0] < -80 and -40 < v[1, 0] < -20 and v[2, 0] > 0
    assert np.all(v[:, 1] == -85.0)


def test_sensor_presets():
    assert ptb49_layout()[0].shape == (49, 2)
    assert kiel36_layout()[0].shape == (36, 2)
    arr = SensorArray(preset="grid", nx=3, ny=2, spacing_mm=10, centre_mm=(5, 0), standoff_mm=10)
    pos, n = arr.place(HalfSpaceSkin(0.05))
    assert pos.shape == (6, 3) and np.allclose(pos[:, 2], 0.06) and np.allclose(n, [0, 0, 1])
    assert np.isclose(pos[:, 0].mean(), 0.005)
