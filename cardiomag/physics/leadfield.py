"""Linear map from nodal membrane voltage to magnetic field at sensors.

Primary current per tetrahedron:  J_e = -sigma_e grad(Vm)  (piecewise-linear Vm -> constant gradient)
Element current dipole:           m_e = vol_e J_e = sum_k a_ek Vm_k,  a_ek = -vol_e sigma_e g_ek
Field component along n_s:        B_s = mu0/4pi sum_e (m_e x R_se)/|R_se|^3 . n_s
                                      = sum_e m_e . L_se,   L_se = mu0/4pi (R_se x n_s)/|R_se|^3
=> B = W @ Vm  with  W[s, k] = sum_{e containing k} L_se . a_ek

For a flat (half-space) chest, the component normal to the chest is exact from the primary current
alone; tangential components are reported without the volume-current term.

Conductivity (effective, calibrated): sigma_il = K_CAL * 0.174 S/m everywhere (uniform mode);
sigma_it from the region's CV ratio, sigma_it = sigma_il (v_t/v_l)^2. K_CAL = 0.33 was calibrated
against the measured PTB P-wave over the four biatrial patients (Analysis/calibrate_sigma.py).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

MU0_4PI = 1e-7
SIGMA_IL_PHYS = 0.174   # S/m (openCARP default g_il)
K_CAL = 0.33            # effective amplitude calibration (see module docstring)
# Ventricular amplitude factor (relative to K_CAL): the simplified fast endocardial layer activates the
# ventricles too synchronously (opposite walls cancel), giving a QRS/P ratio of ~2.7 against 15.6 measured
# (median of PTB + 7 Kiel recordings, a model-free ratio). Effective factor so the ratio matches; with it
# the simulated PTB-layout QRS is ~47 pT vs 48 pT measured.
K_VENT = 5.9


def element_geometry(points, tets):
    """Returns centroids (E,3), volumes (E,), shape-function gradients g (E,4,3)."""
    p0 = points[tets[:, 0]]
    D = np.stack([points[tets[:, i]] - p0 for i in (1, 2, 3)], axis=2)  # columns = edges
    det = np.linalg.det(D)
    vol = np.abs(det) / 6.0
    good = vol > 0
    Minv = np.zeros_like(D)
    Minv[good] = np.linalg.inv(np.transpose(D[good], (0, 2, 1)))  # grad = Minv @ (V1-V0, V2-V0, V3-V0)
    g = np.zeros((tets.shape[0], 4, 3))
    g[:, 1:, :] = np.transpose(Minv, (0, 2, 1))
    g[:, 0, :] = -g[:, 1:, :].sum(axis=1)
    centroid = points[tets].mean(axis=1)
    return centroid, vol, g


def conductivity_tensors(fibres, region, region_names, cv, k_cal=K_CAL, scale=None, region_scale=None):
    """(E,3,3) intracellular conductivity tensors. `scale` (E,) multiplies sigma (e.g. 0 in scar);
    `region_scale` {region name: factor} (e.g. K_VENT for LV/RV)."""
    sl = np.full(region.shape, k_cal * SIGMA_IL_PHYS)
    for idx, name in region_names.items():
        if region_scale and name in region_scale:
            sl[region == idx] *= region_scale[name]
    ratio = np.ones(region.shape)
    for idx, name in region_names.items():
        vl, vt = cv[name]
        ratio[region == idx] = (vt / vl) ** 2
    st = sl * ratio
    if scale is not None:
        sl, st = sl * scale, st * scale
    f = fibres
    return st[:, None, None] * np.eye(3)[None] + (sl - st)[:, None, None] * f[:, :, None] * f[:, None, :]


def element_dipole_operator(points, tets, sigma):
    """a (E,4,3): element dipole moment m_e = sum_k a[e,k] * Vm[tets[e,k]]  (A m per V)."""
    centroid, vol, g = element_geometry(points, tets)
    a = -vol[:, None, None] * np.einsum("eij,ekj->eki", sigma, g)
    return centroid, a


def node_leadfield(points, tets, sigma, sensors, normals, n_nodes=None, chunk=60000, elements=None):
    """W (S, N) [T/V]: B_s = W @ Vm (Vm in volts). `elements` restricts the sum (e.g. scar subtraction)."""
    n_nodes = n_nodes or points.shape[0]
    if elements is not None:
        tets, sigma = tets[elements], sigma[elements]
    centroid, a = element_dipole_operator(points, tets, sigma)
    S = sensors.shape[0]
    W = np.zeros((S, n_nodes))
    for i in range(0, tets.shape[0], chunk):
        sl = slice(i, i + chunk)
        R = sensors[:, None, :] - centroid[None, sl, :]          # (S,Ec,3)
        d3 = np.linalg.norm(R, axis=2) ** 3
        L = MU0_4PI * np.cross(R, normals[:, None, :]) / d3[..., None]  # (S,Ec,3)
        C = np.einsum("sei,eki->sek", L, a[sl])                    # (S,Ec,4)
        Ec = C.shape[1]
        P = sp.csr_matrix((np.ones(4 * Ec), (np.arange(4 * Ec), tets[sl].ravel())), shape=(4 * Ec, n_nodes))
        W += (P.T @ C.reshape(S, 4 * Ec).T).T
    return W


def dipole_operator_total(points, tets, sigma, n_nodes=None):
    """Q (3, N): total current dipole P = Q @ Vm  (A m per V)."""
    n_nodes = n_nodes or points.shape[0]
    _, a = element_dipole_operator(points, tets, sigma)
    Q = np.zeros((3, n_nodes))
    for k in range(4):
        for c in range(3):
            Q[c] += np.bincount(tets[:, k], weights=a[:, k, c], minlength=n_nodes)
    return Q


def voxel_moments(points, tets, sigma, vm_frames, voxel=0.006):
    """Lump element dipoles into voxels for compact export (viewer). Returns centres (V,3), m (T,V,3)."""
    centroid, a = element_dipole_operator(points, tets, sigma)
    key = np.floor(centroid / voxel).astype(np.int64)
    _, vid = np.unique(key, axis=0, return_inverse=True)
    vid = vid.ravel()
    nv = vid.max() + 1
    _, vol, _ = element_geometry(points, tets)
    w = np.bincount(vid, weights=vol, minlength=nv)
    centres = np.stack([np.bincount(vid, weights=centroid[:, c] * vol, minlength=nv) for c in range(3)], 1) / w[:, None]
    m = np.zeros((vm_frames.shape[0], nv, 3))
    for t in range(vm_frames.shape[0]):
        v = vm_frames[t][tets]                                      # (E,4)
        me = np.einsum("eki,ek->ei", a, v)
        for c in range(3):
            m[t, :, c] = np.bincount(vid, weights=me[:, c], minlength=nv)
    return centres, m


def sigma_for(h, scale=None):
    """Conductivity tensors of a HeartMesh, including its per-region factors."""
    return conductivity_tensors(h.fibres, h.region, h.region_names, h.cv, scale=scale,
                                region_scale=getattr(h, "sigma_region_scale", None))
