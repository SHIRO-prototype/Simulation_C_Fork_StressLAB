"""Synthetic conjunction case generation with physics-bounded states.

Produces Stress Tester compatible CaseSnapshot payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from stresslab.stresslab_types import MU_EARTH_KM3S2, RE_EARTH_KM


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-16:
        return np.array([1.0, 0.0, 0.0])
    return v / n


def _random_orthonormal(rng: np.random.Generator, base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    e1 = _unit(base)
    trial = _unit(rng.normal(size=3))
    e2 = trial - np.dot(trial, e1) * e1
    e2 = _unit(e2)
    return e1, e2


def _rtn_basis(state: np.ndarray) -> np.ndarray:
    r = _unit(state[:3])
    h = _unit(np.cross(state[:3], state[3:]))
    t = _unit(np.cross(h, r))
    return np.column_stack([r, t, h])


def _state_from_circular_elements(a_km: float, inc_deg: float, raan_deg: float, u_deg: float) -> np.ndarray:
    inc = np.deg2rad(inc_deg)
    raan = np.deg2rad(raan_deg)
    u = np.deg2rad(u_deg)

    r_pqw = np.array([a_km * np.cos(u), a_km * np.sin(u), 0.0])
    vmag = np.sqrt(MU_EARTH_KM3S2 / a_km)
    v_pqw = np.array([-vmag * np.sin(u), vmag * np.cos(u), 0.0])

    cos_O, sin_O = np.cos(raan), np.sin(raan)
    cos_i, sin_i = np.cos(inc), np.sin(inc)
    R = np.array([
        [cos_O, -sin_O * cos_i, sin_O * sin_i],
        [sin_O, cos_O * cos_i, -cos_O * sin_i],
        [0.0, sin_i, cos_i],
    ])
    pos = R @ r_pqw
    vel = R @ v_pqw
    return np.concatenate([pos, vel])


def _synthesize_covariance(
    rng: np.random.Generator,
    pos_sigma_m: float,
    vel_sigma_mps: float,
    corr_scale: float = 0.08,
) -> np.ndarray:
    pos_sigma_km = pos_sigma_m / 1000.0
    vel_sigma_kms = vel_sigma_mps / 1000.0

    var = np.array([
        pos_sigma_km**2,
        pos_sigma_km**2,
        pos_sigma_km**2,
        vel_sigma_kms**2,
        vel_sigma_kms**2,
        vel_sigma_kms**2,
    ])
    P = np.diag(var)
    for i in range(6):
        for j in range(i + 1, 6):
            rho = float(rng.uniform(-corr_scale, corr_scale))
            cov_ij = rho * np.sqrt(var[i] * var[j])
            P[i, j] = cov_ij
            P[j, i] = cov_ij

    P = 0.5 * (P + P.T)
    eps = 1e-12
    min_eig = float(np.min(np.linalg.eigvalsh(P)))
    while min_eig < 0.0:
        P += (abs(min_eig) + eps) * np.eye(6)
        min_eig = float(np.min(np.linalg.eigvalsh(P)))
    return P


@dataclass(frozen=True)
class TcaPreview:
    tca_seconds: float
    miss_distance_km: float
    rel_velocity_kms: float


def _linear_tca_preview(state1: np.ndarray, state2: np.ndarray, horizon_s: float) -> TcaPreview:
    r = state2[:3] - state1[:3]
    v = state2[3:] - state1[3:]
    vv = float(np.dot(v, v))
    if vv < 1e-16:
        tca = 0.0
    else:
        tca = float(np.clip(-np.dot(r, v) / vv, 0.0, horizon_s))
    miss = float(np.linalg.norm(r + v * tca))
    rel_v = float(np.linalg.norm(v))
    return TcaPreview(tca_seconds=tca, miss_distance_km=miss, rel_velocity_kms=rel_v)


def generate_synthetic_case(
    *,
    case_id: str,
    mode: str = "A",
    seed: int = 42,
    target_miss_m: float = 200.0,
    target_tca_hours: float = 4.0,
    target_rel_speed_mps: float = 10000.0,
    primary_pos_sigma_m: float = 20.0,
    secondary_pos_sigma_m: float = 100.0,
    primary_vel_sigma_mps: float = 0.02,
    secondary_vel_sigma_mps: float = 0.10,
    hard_body_radius_m: float = 5.0,
    horizon_seconds: float = 172800.0,
) -> dict[str, Any]:
    """Generate a physics-bounded synthetic CaseSnapshot plus preview."""
    rng = np.random.default_rng(seed)
    mode_u = str(mode).upper().strip()
    if mode_u not in {"A", "B"}:
        raise ValueError("mode must be A or B")
    if target_miss_m <= 0 or target_tca_hours <= 0 or target_rel_speed_mps <= 0:
        raise ValueError("target_miss_m, target_tca_hours, and target_rel_speed_mps must be > 0")

    epoch_utc = _iso_now()
    tca_s = float(target_tca_hours) * 3600.0
    miss_km = float(target_miss_m) / 1000.0

    attempts = 0
    while True:
        attempts += 1
        if attempts > 40:
            raise RuntimeError("Failed to synthesize physics-bounded conjunction after 40 attempts")

        a = float(RE_EARTH_KM + rng.uniform(450.0, 1100.0))
        inc = float(rng.uniform(0.0, 98.0))
        raan = float(rng.uniform(0.0, 360.0))
        u = float(rng.uniform(0.0, 360.0))
        primary = _state_from_circular_elements(a, inc, raan, u)

        if mode_u == "A":
            basis = _rtn_basis(primary)
            phi = float(rng.uniform(0.0, 2.0 * np.pi))
            rel_speed = float(target_rel_speed_mps / 1000.0)
            v_rtn = np.array([0.0, np.cos(phi), np.sin(phi)]) * rel_speed
            miss_rtn = np.array([0.0, -np.sin(phi), np.cos(phi)]) * miss_km
        else:
            sec = _state_from_circular_elements(
                a * float(1.0 + rng.uniform(-3e-4, 3e-4)),
                inc + float(rng.uniform(-0.2, 0.2)),
                raan + float(rng.uniform(-0.3, 0.3)),
                u + float(rng.uniform(-0.8, 0.8)),
            )
            basis = _rtn_basis(primary)
            rel_vel_seed = sec[3:] - primary[3:]
            v_rtn_seed = basis.T @ rel_vel_seed
            vt = float(v_rtn_seed[1])
            vn = float(v_rtn_seed[2])
            seed_norm = np.hypot(vt, vn)
            if seed_norm < 5e-5:
                phi = float(rng.uniform(0.0, 2.0 * np.pi))
                rel_speed = float(rng.uniform(5e-5, 2e-3))
                v_rtn = np.array([0.0, np.cos(phi), np.sin(phi)]) * rel_speed
            else:
                scale = float(rng.uniform(0.8, 1.2))
                v_rtn = np.array([0.0, vt, vn]) * scale
                rel_speed = float(np.linalg.norm(v_rtn))
                if rel_speed < 5e-5:
                    v_rtn = v_rtn / max(rel_speed, 1e-12) * 5e-5

            tangential = np.array([0.0, v_rtn[1], v_rtn[2]])
            if np.linalg.norm(tangential) < 1e-12:
                tangential = np.array([0.0, 1.0, 0.0])
            tangential = tangential / np.linalg.norm(tangential)
            miss_dir = np.array([0.0, -tangential[2], tangential[1]])
            miss_rtn = miss_dir * miss_km

        # Build secondary state from synthesized relative geometry.
        secondary = primary.copy()
        rel_pos_rtn = miss_rtn - v_rtn * tca_s
        rel_pos = basis @ rel_pos_rtn
        rel_vel = basis @ v_rtn
        secondary[:3] = primary[:3] + rel_pos
        secondary[3:] = primary[3:] + rel_vel

        r1 = float(np.linalg.norm(primary[:3]))
        r2 = float(np.linalg.norm(secondary[:3]))
        v1 = float(np.linalg.norm(primary[3:]))
        v2 = float(np.linalg.norm(secondary[3:]))
        if not (6000.0 <= r1 <= 80000.0 and 6000.0 <= r2 <= 80000.0):
            continue
        if not (0.0 <= v1 <= 20.0 and 0.0 <= v2 <= 20.0):
            continue
        break

    p_cov = _synthesize_covariance(rng, primary_pos_sigma_m, primary_vel_sigma_mps)
    s_cov = _synthesize_covariance(rng, secondary_pos_sigma_m, secondary_vel_sigma_mps)

    preview = _linear_tca_preview(primary, secondary, horizon_s=float(horizon_seconds))

    tca_utc = (
        datetime.fromisoformat(epoch_utc.replace("Z", "+00:00"))
        + timedelta(seconds=preview.tca_seconds)
    ).isoformat().replace("+00:00", "Z")

    case_snapshot = {
        "case_id": case_id,
        "epoch_utc": epoch_utc,
        "primary": {
            "state_eci_km_kms": primary.tolist(),
            "cov_eci_6x6": p_cov.tolist(),
        },
        "secondary": {
            "state_eci_km_kms": secondary.tolist(),
            "cov_eci_6x6": s_cov.tolist(),
        },
        "meta": {
            "generator": "stresslab.synth_case",
            "mode": mode_u,
            "seed": int(seed),
            "hbr_m": float(hard_body_radius_m),
            "target_tca_hours": float(target_tca_hours),
            "target_rel_speed_mps": float(target_rel_speed_mps),
            "propagation_model": "two_body_plus_J2",
            "horizon_seconds": float(horizon_seconds),
        },
    }
    baseline_preview = {
        "estimated_tca_utc": tca_utc,
        "estimated_miss_distance_m": float(preview.miss_distance_km * 1000.0),
        "estimated_rel_vel_mps": float(preview.rel_velocity_kms * 1000.0),
        "initial_separation_km": float(np.linalg.norm(secondary[:3] - primary[:3])),
    }
    return {"case_snapshot": case_snapshot, "preview": baseline_preview}
