from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class RoomConfig:
    lx: float = 5.0
    ly: float = 4.0
    lz: float = 2.7
    nx: int = 30
    ny: int = 24
    nz: int = 18
    nu: float = 1.5e-5
    diffusivity: float = 1.0e-4


@dataclass
class PurifierConfig:
    x: float = 3.0
    y: float = 2.5
    radius: float = 0.11
    height: float = 0.36
    flow_m3s: float = 0.0665
    outlet_radius: float = 0.075
    outlet_z: float = 0.36
    filtration_efficiency: float = 0.97


@dataclass
class SourceConfig:
    x0: float = 1.2
    x1: float = 1.8
    y0: float = 2.8
    y1: float = 3.4
    z0: float = 0.0
    z1: float = 0.2
    emission_rate: float = 1.0e-6


class AirRoomSimulation:
    """Room-scale CFD-lite solver (coarse 3D projection + scalar transport)."""

    def __init__(self, room: RoomConfig, purifier: PurifierConfig, source: SourceConfig):
        self.room = room
        self.purifier = purifier
        self.source = source

        self.dx = room.lx / room.nx
        self.dy = room.ly / room.ny
        self.dz = room.lz / room.nz
        self.cell_vol = self.dx * self.dy * self.dz

        shape = (room.nx, room.ny, room.nz)
        self.u = np.zeros(shape)
        self.v = np.zeros(shape)
        self.w = np.zeros(shape)
        self.p = np.zeros(shape)
        self.c = np.zeros(shape)

        self.solid = np.zeros(shape, dtype=bool)
        self._build_obstacles()
        self._build_source_mask()

        self.last_dt = 0.0
        self.last_sink_mass_rate = 0.0
        self.last_source_mass_rate = source.emission_rate

    def _grid(self):
        x = (np.arange(self.room.nx) + 0.5) * self.dx
        y = (np.arange(self.room.ny) + 0.5) * self.dy
        z = (np.arange(self.room.nz) + 0.5) * self.dz
        return np.meshgrid(x, y, z, indexing="ij")

    def _build_obstacles(self):
        X, Y, Z = self._grid()
        couch = (X > 3.2) & (X < 4.8) & (Y > 0.3) & (Y < 1.2) & (Z < 0.9)
        table = (X > 2.1) & (X < 2.9) & (Y > 1.5) & (Y < 2.3) & (Z < 0.75)
        cabinet = (X > 0.4) & (X < 1.2) & (Y > 0.4) & (Y < 0.9) & (Z < 1.9)
        self.solid |= couch | table | cabinet

    def _build_source_mask(self):
        X, Y, Z = self._grid()
        s = self.source
        self.source_mask = (
            (X >= s.x0) & (X <= s.x1) &
            (Y >= s.y0) & (Y <= s.y1) &
            (Z >= s.z0) & (Z <= s.z1)
        ) & (~self.solid)

    def set_purifier(self, x: float, y: float, flow_m3s: float):
        self.purifier.x = x
        self.purifier.y = y
        self.purifier.flow_m3s = flow_m3s

    def set_source_emission(self, emission_rate: float):
        self.source.emission_rate = emission_rate

    def _apply_no_slip(self):
        self.u[self.solid] = 0.0
        self.v[self.solid] = 0.0
        self.w[self.solid] = 0.0
        for arr in (self.u, self.v, self.w):
            arr[0, :, :] = 0.0
            arr[-1, :, :] = 0.0
            arr[:, 0, :] = 0.0
            arr[:, -1, :] = 0.0
            arr[:, :, 0] = 0.0
            arr[:, :, -1] = 0.0

    def _laplacian(self, f):
        lap = np.zeros_like(f)
        lap[1:-1, 1:-1, 1:-1] = (
            (f[2:, 1:-1, 1:-1] - 2.0 * f[1:-1, 1:-1, 1:-1] + f[:-2, 1:-1, 1:-1]) / self.dx**2
            + (f[1:-1, 2:, 1:-1] - 2.0 * f[1:-1, 1:-1, 1:-1] + f[1:-1, :-2, 1:-1]) / self.dy**2
            + (f[1:-1, 1:-1, 2:] - 2.0 * f[1:-1, 1:-1, 1:-1] + f[1:-1, 1:-1, :-2]) / self.dz**2
        )
        return lap

    def _trilinear(self, q, x, y, z):
        gx = np.clip(x / self.dx - 0.5, 0.0, self.room.nx - 1.001)
        gy = np.clip(y / self.dy - 0.5, 0.0, self.room.ny - 1.001)
        gz = np.clip(z / self.dz - 0.5, 0.0, self.room.nz - 1.001)
        i0 = np.floor(gx).astype(int)
        j0 = np.floor(gy).astype(int)
        k0 = np.floor(gz).astype(int)
        i1 = np.minimum(i0 + 1, self.room.nx - 1)
        j1 = np.minimum(j0 + 1, self.room.ny - 1)
        k1 = np.minimum(k0 + 1, self.room.nz - 1)
        tx = gx - i0
        ty = gy - j0
        tz = gz - k0

        c000 = q[i0, j0, k0]
        c100 = q[i1, j0, k0]
        c010 = q[i0, j1, k0]
        c110 = q[i1, j1, k0]
        c001 = q[i0, j0, k1]
        c101 = q[i1, j0, k1]
        c011 = q[i0, j1, k1]
        c111 = q[i1, j1, k1]

        c00 = c000 * (1 - tx) + c100 * tx
        c10 = c010 * (1 - tx) + c110 * tx
        c01 = c001 * (1 - tx) + c101 * tx
        c11 = c011 * (1 - tx) + c111 * tx

        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty
        return c0 * (1 - tz) + c1 * tz

    def _semi_lagrangian_advect(self, q, dt):
        X, Y, Z = self._grid()
        xb = X - dt * self.u
        yb = Y - dt * self.v
        zb = Z - dt * self.w
        out = self._trilinear(q, xb, yb, zb)
        out[self.solid] = 0.0
        return out

    def _divergence(self):
        div = np.zeros_like(self.u)
        div[1:-1, 1:-1, 1:-1] = (
            (self.u[2:, 1:-1, 1:-1] - self.u[:-2, 1:-1, 1:-1]) / (2 * self.dx)
            + (self.v[1:-1, 2:, 1:-1] - self.v[1:-1, :-2, 1:-1]) / (2 * self.dy)
            + (self.w[1:-1, 1:-1, 2:] - self.w[1:-1, 1:-1, :-2]) / (2 * self.dz)
        )
        div[self.solid] = 0.0
        return div

    def _pressure_projection(self, dt, nit=70):
        rhs = self._divergence() / max(dt, 1e-8)
        p = self.p.copy()
        coef = 2 / self.dx**2 + 2 / self.dy**2 + 2 / self.dz**2

        for _ in range(nit):
            p_old = p.copy()
            p[1:-1, 1:-1, 1:-1] = (
                (p_old[2:, 1:-1, 1:-1] + p_old[:-2, 1:-1, 1:-1]) / self.dx**2
                + (p_old[1:-1, 2:, 1:-1] + p_old[1:-1, :-2, 1:-1]) / self.dy**2
                + (p_old[1:-1, 1:-1, 2:] + p_old[1:-1, 1:-1, :-2]) / self.dz**2
                - rhs[1:-1, 1:-1, 1:-1]
            ) / coef
            p[self.solid] = 0.0
            p[0, :, :] = p[1, :, :]
            p[-1, :, :] = p[-2, :, :]
            p[:, 0, :] = p[:, 1, :]
            p[:, -1, :] = p[:, -2, :]
            p[:, :, 0] = p[:, :, 1]
            p[:, :, -1] = p[:, :, -2]

        self.p = p
        self.u[1:-1, 1:-1, 1:-1] -= dt * (p[2:, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1]) / (2 * self.dx)
        self.v[1:-1, 1:-1, 1:-1] -= dt * (p[1:-1, 2:, 1:-1] - p[1:-1, :-2, 1:-1]) / (2 * self.dy)
        self.w[1:-1, 1:-1, 1:-1] -= dt * (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, :-2]) / (2 * self.dz)

    def _apply_purifier(self, dt):
        X, Y, Z = self._grid()
        p = self.purifier
        r = np.sqrt((X - p.x) ** 2 + (Y - p.y) ** 2)

        outlet_mask = (r <= p.outlet_radius + 0.5*self.dx) & (np.abs(Z - p.outlet_z) <= 1.5*self.dz)
        outlet_area = np.pi * p.outlet_radius**2
        uz = p.flow_m3s / max(outlet_area, 1e-6)
        self.w[outlet_mask] = 0.8 * self.w[outlet_mask] + 0.2 * uz

        shell = (r >= max(p.radius - 0.03, 0.02)) & (r <= p.radius + 0.12) & (Z >= 0.03) & (Z <= p.height + 0.05)
        if shell.any():
            rx = p.x - X[shell]
            ry = p.y - Y[shell]
            rr = np.sqrt(rx**2 + ry**2) + 1e-8
            intake_area = shell.sum() * self.dy * self.dz
            vin = min(p.flow_m3s / max(intake_area, 1e-6), 1.2)
            ux = vin * rx / rr
            vy = vin * ry / rr
            self.u[shell] = 0.8 * self.u[shell] + 0.2 * ux
            self.v[shell] = 0.8 * self.v[shell] + 0.2 * vy

            k = p.filtration_efficiency * p.flow_m3s / max(shell.sum() * self.cell_vol, 1e-9)
            c_before = self.c[shell].copy()
            self.c[shell] *= np.exp(-k * dt)
            removed = (c_before - self.c[shell]).sum() * self.cell_vol
            self.last_sink_mass_rate = removed / max(dt, 1e-8)
        else:
            self.last_sink_mass_rate = 0.0

    def _apply_source(self, dt):
        src_vol = self.source_mask.sum() * self.cell_vol
        self.last_source_mass_rate = self.source.emission_rate
        if src_vol > 0:
            self.c[self.source_mask] += (self.source.emission_rate / src_vol) * dt

    def _diffuse(self, dt):
        self.u += dt * self.room.nu * self._laplacian(self.u)
        self.v += dt * self.room.nu * self._laplacian(self.v)
        self.w += dt * self.room.nu * self._laplacian(self.w)
        self.c += dt * self.room.diffusivity * self._laplacian(self.c)

    def _stable_dt(self, dt_requested, cfl_target=0.45):
        speed = np.sqrt(self.u**2 + self.v**2 + self.w**2)
        vmax = float(np.max(speed[~self.solid])) if np.any(~self.solid) else 0.0
        h = min(self.dx, self.dy, self.dz)
        dt_cfl = cfl_target * h / max(vmax, 1e-6)
        dt_diff = 0.3 * h**2 / max(self.room.diffusivity, self.room.nu, 1e-8)
        return float(max(min(dt_requested, dt_cfl, dt_diff), 1e-4))

    def step(self, dt_requested):
        dt = self._stable_dt(dt_requested)
        self.last_dt = dt

        self.u = self._semi_lagrangian_advect(self.u, dt)
        self.v = self._semi_lagrangian_advect(self.v, dt)
        self.w = self._semi_lagrangian_advect(self.w, dt)

        self._diffuse(dt)
        self._apply_purifier(dt)
        self._apply_no_slip()
        self._pressure_projection(dt)
        self._apply_no_slip()

        self.c = self._semi_lagrangian_advect(self.c, dt)
        self.c += dt * self.room.diffusivity * self._laplacian(self.c)
        self._apply_source(dt)
        self.c[self.solid] = 0.0
        self.c = np.clip(self.c, 0.0, None)
        return dt

    def diagnostics(self, dt_requested) -> Dict[str, float]:
        div = self._divergence()
        speed = np.sqrt(self.u**2 + self.v**2 + self.w**2)
        vmax = float(np.max(speed[~self.solid])) if np.any(~self.solid) else 0.0
        dt_use = self.last_dt if self.last_dt > 0 else dt_requested
        cfl = vmax * dt_use / min(self.dx, self.dy, self.dz)
        mass = float(np.sum(self.c[~self.solid]) * self.cell_vol)

        # mass-balance residual: dM/dt - source + sink
        # approximate dM/dt over one step using current rate estimate only
        mb_res = self.last_source_mass_rate - self.last_sink_mass_rate

        return {
            "dt_used": float(dt_use),
            "max_divergence": float(np.max(np.abs(div[~self.solid]))),
            "mean_divergence": float(np.mean(np.abs(div[~self.solid]))),
            "max_velocity": vmax,
            "cfl": float(cfl),
            "total_pollutant_mass": mass,
            "source_mass_rate": float(self.last_source_mass_rate),
            "sink_mass_rate": float(self.last_sink_mass_rate),
            "net_input_rate": float(mb_res),
        }

    def reset(self):
        self.u.fill(0.0)
        self.v.fill(0.0)
        self.w.fill(0.0)
        self.p.fill(0.0)
        self.c.fill(0.0)
        self.last_dt = 0.0
        self.last_sink_mass_rate = 0.0
        self.last_source_mass_rate = self.source.emission_rate
