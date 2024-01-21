""" 
freq_analysis.py

Use the frequency domain beamforming method to estimate the bf parameters.
"""
from typing import Tuple

import numpy as np
import pymap3d as pm
from numba import complex128, float64, int64, njit, objmode
from numba.experimental import jitclass
from numpy.typing import NDArray


@njit()
def parallel_get_weight_omega(self, omega_idx: int):
    # not parallel anyway as we are using MPI
    # change from 6,6,0.3 to 2,2,0.1 will not change the result
    phis = np.arange(-90, 90, 6)
    thetas = np.arange(0, 360, 6)
    vs = np.arange(5.5, 11.5, 0.3)

    res = np.zeros((len(phis), len(thetas), len(vs)), dtype=np.float64)
    for i in range(len(phis)):
        phi = phis[i]
        for j, theta in enumerate(thetas):
            for k, v in enumerate(vs):
                res[i, j, k] = self.get_power_s_omega(phi, theta, v, omega_idx)
    return np.max(res)


@jitclass(
    [
        ("waveforms", float64[:, :]),
        ("coordinates_enu", float64[:, :]),
        ("dt", float64),
        ("parse_omega_ratio", float64),
        ("omega", float64[:]),
        ("fft_w", complex128[:, :]),
        ("weighting", float64[:]),
        ("omega_l", int64),
        ("omega_r", int64),
    ]
)
class FreqBF:
    def __init__(
        self,
        waveforms: NDArray[np.float64],
        coordinates_lld: NDArray[np.float64],
        dt: float,
        parse_omega_ratio: float,
    ):
        # * global parameters
        # m: number of traces, n: trace length
        m, n = waveforms.shape
        self.waveforms = waveforms
        self.dt = dt
        self.parse_omega_ratio = parse_omega_ratio

        # * frequency domain data
        df = 1 / dt / n
        self.omega = np.arange(0, n // 2) * df * 2 * np.pi
        self.fft_w = np.zeros((m, n // 2), dtype=np.complex128)
        self.calculate_fft()

        # * coordinates
        self.coordinates_enu = np.zeros((m, 3), dtype=np.float64)
        self.coordinate_conversion(coordinates_lld)

        # * weighting
        stacked_fft_w = np.sum(np.abs(self.fft_w), axis=0) / m
        self.omega_l, self.omega_r = self.parse_omega_range(stacked_fft_w)
        self.weighting = np.zeros((self.omega_r - self.omega_l), dtype=np.float64)
        for i, omega_idx in enumerate(range(self.omega_l, self.omega_r)):
            self.weighting[i] = self.get_weight_omega(omega_idx)

    def calculate_fft(self):
        m, n = self.waveforms.shape
        for itrace in range(m):
            with objmode(fft_w_single_trace="complex128[:]"):
                fft_w_single_trace = np.fft.fft(self.waveforms[itrace, :])[: n // 2] / n
            self.fft_w[itrace, :] = fft_w_single_trace[:]

    def parse_omega_range(self, stacked_fft_w: NDArray[np.float64]) -> Tuple[int, int]:
        fft = np.abs(stacked_fft_w)
        max_fft = np.max(fft)
        max_id = np.argmax(fft)
        l, r = max_id, max_id
        while l > 0 and fft[l] >= self.parse_omega_ratio * max_fft:
            l -= 1
        while r < len(fft) - 1 and fft[r] >= self.parse_omega_ratio * max_fft:
            r += 1
        return l, r

    def coordinate_conversion(self, coordinates_lld: NDArray[np.float64]):
        m, _ = coordinates_lld.shape
        ref_lat = np.mean(coordinates_lld[:, 0])
        ref_lon = np.mean(coordinates_lld[:, 1])
        ref_dep = np.mean(coordinates_lld[:, 2])

        for icoor in range(m):
            with objmode(e="float64", n="float64", u="float64"):
                lat = coordinates_lld[icoor, 0]
                lon = coordinates_lld[icoor, 1]
                dep = coordinates_lld[icoor, 2]
                e, n, u = pm.geodetic2enu(
                    lat, lon, -dep * 1000, ref_lat, ref_lon, -ref_dep * 1000
                )
            self.coordinates_enu[icoor, :] = [e / 1000, n / 1000, u / 1000]

    @staticmethod
    def velocity_to_slowness(
        phi: float, theta: float, v: float
    ) -> Tuple[np.float64, np.float64, np.float64]:
        p, t = np.deg2rad(phi), np.deg2rad(theta)
        sx = np.cos(p) * np.cos(t) / v
        sy = np.cos(p) * np.sin(t) / v
        sz = np.sin(p) / v
        return sx, sy, sz

    def get_u_s_omega(
        self, phi: float, theta: float, v: float, omega_idx: int
    ) -> np.complex128:
        m, _ = np.shape(self.coordinates_enu)
        res = 0 + 0j
        sx, sy, sz = self.velocity_to_slowness(phi, theta, v)
        for i in range(m):
            r = self.coordinates_enu[i]
            omega = self.omega[omega_idx]
            res += (
                np.exp(omega * (sx * r[0] + sy * r[1] + sz * r[2]) * 1j)
                * self.fft_w[i, omega_idx]
            )
        return res

    def get_power_s_omega(
        self, phi: float, theta: float, v: float, omega_idx: int
    ) -> float:
        u = self.get_u_s_omega(phi, theta, v, omega_idx)
        return np.real(u * np.conjugate(u))

    def get_weight_omega(self, omega_idx: int) -> float:
        return parallel_get_weight_omega(self, omega_idx)

    def get_power_s(self, phi: float, theta: float, v: float) -> float:
        res = 0
        for i, omega_idx in enumerate(range(self.omega_l, self.omega_r)):
            p = self.get_power_s_omega(phi, theta, v, omega_idx)
            res += p / self.weighting[i]
        res /= self.omega_r - self.omega_l
        return res

    def opt_func(self, parameter: Tuple[float, float, float]) -> float:
        phi, theta, v = parameter
        return self.get_power_s(phi, theta, v)
