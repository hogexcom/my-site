"""
Level Set Flow Solver (Single-Phase Free Surface)
Implemented based on the user's requirements:
- Level Set Advection: WENO5 + TVD-RK3
- Reinitialization: Godunov Scheme
- Velocity Extrapolation: Upwind PDE
- Pressure BC: Cheng et al. (2nd order)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import copy
import time

from WENO2D import hj_weno5_derivatives_2d

class LevelSetFlowSolver:
    def __init__(
        self,
        nx: int = 50,
        ny: int = 50,
        re: float = 1000.0,
        dt: float = 0.001,
        lx: float = 1.0,
        ly: float = 1.0,
        initial_condition: str = "flat",
        reinit_mode: str = "interval",
        body_force: Tuple[float, float] = (0.0, -9.8),
    ):
        self.nx = nx
        self.ny = ny
        self.re = re
        self.dt = dt
        self.lx = lx
        self.ly = ly
        self.reinit_mode = reinit_mode

        self.body_force = body_force # (fx, fy)
        self.dx = self.lx / nx
        self.dy = self.ly / ny
        self.nu = 1.0 / re
        
        # Initialize step counter
        self.step_count = 0
        self.reinit_interval = 10
        self.phi_cap = float(np.hypot(self.lx, self.ly))

        # Performance Logging
        self.perf_log = {
            "Tentative Velocity": [],
            "Pressure Solve": [],
            "Correction": [],
            "Level Set Advect": [],
            "Level Set Reinit": [],
            "Velocity Extrapolation": [],
            "Total Step": []
        }
        # u: (ny, nx+1), v: (ny+1, nx)
        self.u = np.zeros((ny, nx + 1), dtype=np.float64)
        self.v = np.zeros((ny + 1, nx), dtype=np.float64)
        
        # Colocated Grid for Pressure and Level Set
        # p: (ny, nx), phi: (ny, nx)
        self.p = np.zeros((ny, nx), dtype=np.float64)

        # Grid setup (Staggered grid logic)
        # Cell centers
        x = np.linspace(self.dx / 2, self.lx - self.dx / 2, nx)
        y = np.linspace(self.dy / 2, self.ly - self.dy / 2, ny)
        self.X, self.Y = np.meshgrid(x, y)

        # Initialize Level Set
        if initial_condition == "flat":
            # Bottom half water
            # Water: phi < 0, Air: phi > 0
            self.phi = self.Y - 0.5 * self.ly
        elif initial_condition == "column":
            # Water Column (Dam Break)
            # Rectangular column: x in [0, 0.4], y in [0, 0.6]
            self.phi = np.maximum(self.X - 0.4, self.Y - 0.6)
        elif initial_condition == "empty":
            # Totally empty (Air everywhere)
            self.phi = np.full((ny, nx), 1.0) # Dist = 1.0 (arbitrary pos)
        else:
            raise ValueError(f"Unknown initial condition: {initial_condition}")


    def _get_advection_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get velocity at cell centers for advection, with extrapolation at boundaries.
        We overwrite the wall BC (no-penetration) with extrapolation to allow 
        the level set interface to approach/cross the wall (wetting).
        """
        u = self.u.copy()
        v = self.v.copy()
        
        # Extrapolate Normal Velocities at Boundaries (Neumann 0 approach)
        # Left/Right
        u[:, 0] = u[:, 1]
        u[:, -1] = u[:, -2]
        
        # Bottom/Top
        v[0, :] = v[1, :]
        v[-1, :] = v[-2, :]
        
        # Interpolate to centers
        u_c = 0.5 * (u[:, :-1] + u[:, 1:])
        v_c = 0.5 * (v[:-1, :] + v[1:, :])
        
        return u_c, v_c

    def _rk3_step(self, phi: np.ndarray, u: np.ndarray, v: np.ndarray, dt: float) -> np.ndarray:
        """One step of TVD-RK3."""
        # Function to compute RHS = - (u * phi_x + v * phi_y)
        def compute_rhs(phi_curr):
            derivs = hj_weno5_derivatives_2d(phi_curr.tolist(), self.dx, self.dy, periodic=(False, False)) # Assume non-periodic for wall bounded
            
            # Upwinding based on velocity
            # Note: u and v here are at cell centers
            phi_x = np.where(u > 0, np.array(derivs.phi_x_minus), np.array(derivs.phi_x_plus))
            phi_y = np.where(v > 0, np.array(derivs.phi_y_minus), np.array(derivs.phi_y_plus))
            
            return -(u * phi_x + v * phi_y)

        # Step 1
        rhs1 = compute_rhs(phi)
        phi1 = phi + dt * rhs1

        # Step 2
        rhs2 = compute_rhs(phi1)
        phi2 = 0.75 * phi + 0.25 * (phi1 + dt * rhs2)

        # Step 3
        rhs3 = compute_rhs(phi2)
        phi_new = (1.0 / 3.0) * phi + (2.0 / 3.0) * (phi2 + dt * rhs3)
        
        return phi_new

    def _apply_contact_angle(self):
        """
        Enforce contact angle BC for Level Set function phi.
        Homogeneous Neumann (dphi/dn = 0) corresponds to 90 degree contact angle.
        This allows the interface to slide along the wall (moving contact line).
        """
        # Left/Right Walls (Neumann)
        self.phi[:, 0] = self.phi[:, 1]
        self.phi[:, -1] = self.phi[:, -2]
        
        # Bottom/Top Walls (Neumann)
        self.phi[0, :] = self.phi[1, :]
        self.phi[-1, :] = self.phi[-2, :]

    def advect_levelset(self):
        """Advect magnitude phi using WENO5 + TVD-RK3."""
        # For advection, we use the extrapolated velocity to allow wetting
        u_c, v_c = self._get_advection_velocity()
        
        self.phi = self._rk3_step(self.phi, u_c, v_c, self.dt)
        self.phi = np.clip(self.phi, -self.phi_cap, self.phi_cap)
        self._apply_contact_angle() # Enforce BC after advection

    def reinitialize_levelset(self, iterations: int = 10):
        """
        Reinitialize phi to be a signed distance function using Godunov scheme.
        dphi/dtau + S(phi0) * (|grad(phi)| - 1) = 0
        """
        phi0 = self.phi.copy()
        # S(phi0) - smoothed sign function
        sign_phi0 = phi0 / np.sqrt(phi0**2 + self.dx**2)
        
        # dtau should be small, e.g., 0.5 * dx
        dtau = 0.5 * self.dx
        
        phi = phi0.copy()
        
        for _ in range(iterations):
            # Compute gradients using central differences (simplified for Godunov input) produces D+ and D-
            # But Godunov requires one-sided differences.
            # D_x^- phi[i] = (phi[i] - phi[i-1]) / dx
            # D_x^+ phi[i] = (phi[i+1] - phi[i]) / dx
            
            # We enforce Neumann 0 (Contact Angle 90) at boundaries:
            # dphi/dn = 0 => Derivative is 0 at the wall.
            
            # --- X Derivatives ---
            phi_xm = np.zeros_like(phi) # D_x^-
            phi_xp = np.zeros_like(phi) # D_x^+
            
            phi_xm[:, 1:] = (phi[:, 1:] - phi[:, :-1]) / self.dx
            phi_xm[:, 0] = 0.0 # Neumann 0 at Left Wall
            
            phi_xp[:, :-1] = (phi[:, 1:] - phi[:, :-1]) / self.dx
            phi_xp[:, -1] = 0.0 # Neumann 0 at Right Wall
            
            # --- Y Derivatives ---
            phi_ym = np.zeros_like(phi) # D_y^-
            phi_yp = np.zeros_like(phi) # D_y^+
            
            phi_ym[1:, :] = (phi[1:, :] - phi[:-1, :]) / self.dy
            phi_ym[0, :] = 0.0 # Neumann 0 at Bottom Wall
            
            phi_yp[:-1, :] = (phi[1:, :] - phi[:-1, :]) / self.dy
            phi_yp[-1, :] = 0.0 # Neumann 0 at Top Wall
            
            # Godunov Selection
            # Initial Sign > 0
            grad_phi_sq_pos = (
                np.maximum(np.maximum(phi_xm, 0)**2, np.minimum(phi_xp, 0)**2) +
                np.maximum(np.maximum(phi_ym, 0)**2, np.minimum(phi_yp, 0)**2)
            )
            
            # Initial Sign < 0
            grad_phi_sq_neg = (
                np.maximum(np.maximum(phi_xp, 0)**2, np.minimum(phi_xm, 0)**2) +
                np.maximum(np.maximum(phi_yp, 0)**2, np.minimum(phi_ym, 0)**2)
            )
            
            grad_phi = np.zeros_like(phi)
            mask_pos = (sign_phi0 > 0)
            mask_neg = (sign_phi0 < 0) # strictly less, =0 handled implicitly or by one of them
            
            grad_phi[mask_pos] = np.sqrt(grad_phi_sq_pos[mask_pos])
            grad_phi[mask_neg] = np.sqrt(grad_phi_sq_neg[mask_neg])
            # For sign_phi0 == 0, grad_phi doesn't strictly matter as coeff is 0, but good to keep bounded
            
            # Update
            # phi_new = phi - dtau * S(phi0) * (grad_phi - 1)
            phi = phi - dtau * sign_phi0 * (grad_phi - 1.0)
            
            # Update self.phi temporarily to apply BC helper (or just manual)
            self.phi = phi
            self._apply_contact_angle()
            phi = self.phi
            
        self.phi = phi
        self.phi = np.clip(self.phi, -self.phi_cap, self.phi_cap)

    def extrapolate_velocity(self, u_in: np.ndarray = None, v_in: np.ndarray = None, iterations: int = 10):
        """
        Extrapolate velocity from liquid to air.
        du/dtau + N . grad(u) = 0, where N = grad(phi)/|grad(phi)|
        Uses first order upwind scheme.
        If u_in, v_in are provided, extrapolates those (returns new arrays).
        Otherwise modifies self.u, self.v in place (and returns them).
        """
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        
        # Determine targets
        if u_in is None:
            u = self.u
            v = self.v
        else:
            u = u_in.copy()
            v = v_in.copy()
        
        # Pseudo time step
        dtau = 0.5 * min(dx, dy)

        # Get phi at cell centers (it already is)
        phi = self.phi

        # Compute Normal vector N = grad(phi) / |grad(phi)| at cell centers
        # User specified logic: Use upwind/characteristic direction (from smaller phi to larger phi)
        # Check neighbors. If neighbor < center, it's a candidate "source" of info.
        # If both neighbors < center, pick the smaller one (deeper in water).
        # If both neighbors > center, no source, grad = 0.
        
        # --- Phi Gradient (X) ---
        phi_x = np.zeros_like(phi)
        
        # Inner domain
        phi_c = phi[:, 1:-1]
        phi_L = phi[:, :-2]
        phi_R = phi[:, 2:]
        
        # dx terms
        dx_L = (phi_c - phi_L) / dx # Backward diff
        dx_R = (phi_R - phi_c) / dx # Forward diff
        
        # Conditions
        use_L = (phi_L < phi_c)
        use_R = (phi_R < phi_c)
        
        # If both, compare phi_L and phi_R
        better_L = use_L & use_R & (phi_L < phi_R)
        better_R = use_L & use_R & (phi_R <= phi_L)
        
        # Exclusive or refined masks
        final_use_L = (use_L & ~use_R) | better_L
        final_use_R = (use_R & ~use_L) | better_R
        
        # Apply
        phi_x[:, 1:-1] = np.where(final_use_L, dx_L, 
                                  np.where(final_use_R, dx_R, 0.0))
                                  
        # Boundaries
        if self.nx > 1:
            # i=0
            mask_b0 = (phi[:, 1] < phi[:, 0])
            phi_x[mask_b0, 0] = (phi[mask_b0, 1] - phi[mask_b0, 0]) / dx
            
            # i=-1
            mask_b1 = (phi[:, -2] < phi[:, -1])
            phi_x[mask_b1, -1] = (phi[mask_b1, -1] - phi[mask_b1, -2]) / dx

        # --- Phi Gradient (Y) ---
        phi_y = np.zeros_like(phi)
        
        # Inner domain
        phi_c = phi[1:-1, :]
        phi_D = phi[:-2, :] # Down (j-1)
        phi_U = phi[2:, :]  # Up (j+1)
        
        # dy terms
        dy_D = (phi_c - phi_D) / dy # Backward
        dy_U = (phi_U - phi_c) / dy # Forward
        
        use_D = (phi_D < phi_c)
        use_U = (phi_U < phi_c)
        
        better_D = use_D & use_U & (phi_D < phi_U)
        better_U = use_D & use_U & (phi_U <= phi_D)
        
        final_use_D = (use_D & ~use_U) | better_D
        final_use_U = (use_U & ~use_D) | better_U
        
        phi_y[1:-1, :] = np.where(final_use_D, dy_D,
                                  np.where(final_use_U, dy_U, 0.0))
        
        # Boundaries
        if self.ny > 1:
            # j=0
            mask_b0 = (phi[1, :] < phi[0, :])
            phi_y[0, mask_b0] = (phi[1, mask_b0] - phi[0, mask_b0]) / dy
                 
            # j=-1
            mask_b1 = (phi[-2, :] < phi[-1, :])
            phi_y[-1, mask_b1] = (phi[-1, mask_b1] - phi[-2, mask_b1]) / dy

        norm = np.sqrt(phi_x**2 + phi_y**2) + 1e-12
        nx_c = phi_x / norm
        ny_c = phi_y / norm
        
        # We extrapolate u and v separately.
        # u is at (ny, nx+1). We interpolate N to u-locations.
        # v is at (ny+1, nx). We interpolate N to v-locations.

        nx_u = 0.5 * (np.pad(nx_c, ((0, 0), (1, 1)), 'edge')[:, :-1] + np.pad(nx_c, ((0, 0), (1, 1)), 'edge')[:, 1:]) 
        ny_u = 0.5 * (np.pad(ny_c, ((0, 0), (1, 1)), 'edge')[:, :-1] + np.pad(ny_c, ((0, 0), (1, 1)), 'edge')[:, 1:])
        
        nx_v = 0.5 * (np.pad(nx_c, ((1, 1), (0, 0)), 'edge')[:-1, :] + np.pad(nx_c, ((1, 1), (0, 0)), 'edge')[1:, :])
        ny_v = 0.5 * (np.pad(ny_c, ((1, 1), (0, 0)), 'edge')[:-1, :] + np.pad(ny_c, ((1, 1), (0, 0)), 'edge')[1:, :])

        # To respect the condition that we only extrapolate FROM liquid TO air,
        # we can use a mask or just run it everywhere and let the characteristics carry info outward.
        # Since Grad(phi) points to Air, N points to Air.
        # So we transport u along N.
        
        # u, v are already set from inputs or self

        # Mask: we only update points where phi > 0 (Air)
        # We need phi at u and v locations
        phi_u = 0.5 * (np.pad(phi, ((0, 0), (1, 1)), 'edge')[:, :-1] + np.pad(phi, ((0, 0), (1, 1)), 'edge')[:, 1:])
        phi_v = 0.5 * (np.pad(phi, ((1, 1), (0, 0)), 'edge')[:-1, :] + np.pad(phi, ((1, 1), (0, 0)), 'edge')[1:, :])
        
        mask_u = (phi_u > 0)
        mask_v = (phi_v > 0)

        for _ in range(iterations):
            # Upwind derivatives for u
            # du/dx: if nx_u > 0, use u[i] - u[i-1], else u[i+1] - u[i]
            u_xp = np.zeros_like(u)
            u_xm = np.zeros_like(u)
            u_xp[:, :-1] = (u[:, 1:] - u[:, :-1]) / dx
            u_xp[:, -1]  = 0 # Boundary
            u_xm[:, 1:] = (u[:, 1:] - u[:, :-1]) / dx
            u_xm[:, 0]  = 0
            
            u_dx = np.where(nx_u > 0, u_xm, u_xp)
            
            # du/dy
            u_yp = np.zeros_like(u)
            u_ym = np.zeros_like(u)
            u_yp[:-1, :] = (u[1:, :] - u[:-1, :]) / dy
            u_ym[1:, :] = (u[1:, :] - u[:-1, :]) / dy
            
            u_dy = np.where(ny_u > 0, u_ym, u_yp)
            
            grad_u_dot_n = nx_u * u_dx + ny_u * u_dy
            
            # Update only in air
            u[mask_u] = u[mask_u] - dtau * grad_u_dot_n[mask_u]

            # Upwind derivatives for v
            v_xp = np.zeros_like(v)
            v_xm = np.zeros_like(v)
            v_xp[:, :-1] = (v[:, 1:] - v[:, :-1]) / dx
            v_xm[:, 1:] = (v[:, 1:] - v[:, :-1]) / dx
            
            v_dx = np.where(nx_v > 0, v_xm, v_xp)
            
            v_yp = np.zeros_like(v)
            v_ym = np.zeros_like(v)
            v_yp[:-1, :] = (v[1:, :] - v[:-1, :]) / dy
            v_ym[1:, :] = (v[1:, :] - v[:-1, :]) / dy
            
            v_dy = np.where(ny_v > 0, v_ym, v_yp)
            
            grad_v_dot_n = nx_v * v_dx + ny_v * v_dy
            
            v[mask_v] = v[mask_v] - dtau * grad_v_dot_n[mask_v]
            
        if u_in is None:
            self.u = u
            self.v = v
            
        return u, v

    def _pad_field(self, f, axis, symmetric=False):
        """Kawamura-Kuwaharaスキーム用にパディングを行う (symmetric=TrueでFree Slip対応)"""
        pad_width = [(0, 0), (0, 0)]
        pad_width[axis] = (2, 2)
        f_pad = np.pad(f, pad_width, mode="constant")

        # Multiplier: -1 if anti-symmetric (Dirichlet 0), 1 if symmetric (Neumann 0)
        m = 1.0 if symmetric else -1.0

        if axis == 0:  # y-padding
            f_pad[1, :] = m * f_pad[2, :]
            f_pad[0, :] = m * f_pad[3, :]
            f_pad[-2, :] = m * f_pad[-3, :]
            f_pad[-1, :] = m * f_pad[-4, :]
        else:  # x-padding
            f_pad[:, 1] = m * f_pad[:, 3]
            f_pad[:, 0] = m * f_pad[:, 4]
            f_pad[:, -2] = m * f_pad[:, -4]
            f_pad[:, -1] = m * f_pad[:, -5]
        return f_pad

    def _kawamura_kuwahara(self, f_pad, u_pad, h, axis):
        """Kawamura-Kuwaharaスキーム (3次精度風上差分)"""
        if axis == 1:  # x-axis
            f_m2 = f_pad[:, :-4]
            f_m1 = f_pad[:, 1:-3]
            f_0 = f_pad[:, 2:-2]
            f_p1 = f_pad[:, 3:-1]
            f_p2 = f_pad[:, 4:]
            u_0 = u_pad[:, 2:-2]
        else:  # y-axis
            f_m2 = f_pad[:-4, :]
            f_m1 = f_pad[1:-3, :]
            f_0 = f_pad[2:-2, :]
            f_p1 = f_pad[3:-1, :]
            f_p2 = f_pad[4:, :]
            u_0 = u_pad[2:-2, :]
            u_0 = u_pad[2:-2, :]
        
        # 4th order central difference
        diff_central = (-f_p2 + 8 * f_p1 - 8 * f_m1 + f_m2) / (12 * h)

        # 4th order dissipation (coefficient |u|/4)
        diff_dissip = (f_p2 - 4 * f_p1 + 6 * f_0 - 4 * f_m1 + f_m2) / (4 * h)

        return u_0 * diff_central + np.abs(u_0) * diff_dissip

    def compute_tentative_velocity(self):
        """粘性項と移流項を計算し、仮の速度場(u_star, v_star)を求める。重力含む。"""
        u, v = self.u, self.v
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        nu = self.nu
        dt = self.dt
        
        # --- Advection (Same as NavierStokesSolver) ---
        # 1. u * du/dx
        # Wall Normal (x): Anti-symmetric (No Penetration)
        u_pad_x = self._pad_field(u, axis=1, symmetric=False)
        adv_u_x = self._kawamura_kuwahara(u_pad_x, u_pad_x, dx, axis=1)[:, 1:-1]

        # 2. v * du/dy
        # Wall Tangential (y): Symmetric (Free Slip)
        u_pad_y = self._pad_field(u, axis=0, symmetric=True)
        
        v_avg = 0.25 * (v[:-1, 1:] + v[:-1, :-1] + v[1:, 1:] + v[1:, :-1])
        v_avg_pad_y = self._pad_field(v_avg, axis=0, symmetric=True) # Matches u padding
        adv_u_y = self._kawamura_kuwahara(u_pad_y[:, 1:-1], v_avg_pad_y, dy, axis=0)

        # 3. u * dv/dx
        # Wall Tangential (x): Symmetric (Free Slip)
        v_pad_x = self._pad_field(v, axis=1, symmetric=True)
        u_avg = 0.25 * (u[1:, :-1] + u[1:, 1:] + u[:-1, :-1] + u[:-1, 1:])
        u_avg_pad_x = self._pad_field(u_avg, axis=1, symmetric=True) # Matches v padding
        adv_v_x = self._kawamura_kuwahara(v_pad_x[1:-1, :], u_avg_pad_x, dx, axis=1)

        # 4. v * dv/dy
        # Wall Normal (y): Anti-symmetric (No Penetration)
        v_pad_y = self._pad_field(v, axis=0, symmetric=False)
        adv_v_y = self._kawamura_kuwahara(v_pad_y, v_pad_y, dy, axis=0)[1:-1, :]

        # --- Viscosity (Same as NavierStokesSolver) ---
        u_star = u.copy()
        
        # u diffusion
        u_xx = (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / dx**2
        
        u_yy = np.zeros_like(u[:, 1:-1])
        u_yy[1:-1, :] = (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dy**2
        # BC: u_y = 0 (Free Slip) at walls y=0, y=H
        # ghost u_{-1} = u_{0} -> u_yy_0 ~ (u_1 - 2u_0 + u_0) = (u_1 - u_0)
        u_yy[0, :] = (u[1, 1:-1] - u[0, 1:-1]) / dy**2
        u_yy[-1, :] = (-u[-1, 1:-1] + u[-2, 1:-1]) / dy**2
        
        u_star[:, 1:-1] = u[:, 1:-1] + dt * (nu * (u_xx + u_yy) - (adv_u_x + adv_u_y))
        
        # v diffusion
        v_star = v.copy()
        
        v_yy = (v[2:, :] - 2 * v[1:-1, :] + v[:-2, :]) / dy**2
        
        v_xx = np.zeros_like(v[1:-1, :])
        v_xx[:, 1:-1] = (v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, :-2]) / dx**2
        # BC: v_x = 0 (Free Slip) at walls x=0, x=L
        v_xx[:, 0] = (v[1:-1, 1] - v[1:-1, 0]) / dx**2
        v_xx[:, -1] = (-v[1:-1, -1] + v[1:-1, -2]) / dx**2

        # Add Body Force terms
        gx, gy = self.body_force
        u_star[:, 1:-1] += dt * gx
        v_star[1:-1, :] += dt * (nu * (v_xx + v_yy) - (adv_v_x + adv_v_y) + gy)

        # Non-Incremental: No pressure gradient in predictor
        
        return u_star, v_star


    def compute_pressure_gradient(self, p, phi):
        """Compute pressure gradient (Cheng's BC aware). Used for both predictor and corrector."""
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        
        grad_p_x = np.zeros_like(self.u)
        grad_p_y = np.zeros_like(self.v)
        
        # --- U Gradient ---
        # i goes from 1 to nx-1
        p_curr = p # p[j, i] (Right)
        p_prev = np.roll(p, 1, axis=1) # p[j, i-1] (Left)
        
        phi_curr = phi
        phi_prev = np.roll(phi, 1, axis=1)
        
        # Masks (Slices)
        # We process inner U-faces: u[j, 1] to u[j, nx-1]
        mask_u_inner = np.zeros_like(self.u, dtype=bool)
        mask_u_inner[:, 1:-1] = True
        
        # Neighbor Level Set Signs
        phi_L = phi_prev
        phi_R = phi_curr
        
        # Conditions
        is_LL = (phi_L < 0) & (phi_R < 0)
        is_LA = (phi_L < 0) & (phi_R > 0)
        is_AL = (phi_L > 0) & (phi_R < 0)
        
        # Standard Gradient (Liquid-Liquid)
        grad_standard = (p_curr - p_prev) / dx
        
        # Interface Gradient (Liquid-Air)
        theta_la = abs(phi_L) / (abs(phi_L) + abs(phi_R) + 1e-12)
        theta_la = np.maximum(theta_la, 1e-2)
        grad_la = (0.0 - p_prev) / (theta_la * dx)
        
        # Interface Gradient (Air-Liquid)
        theta_al = abs(phi_R) / (abs(phi_R) + abs(phi_L) + 1e-12)
        theta_al = np.maximum(theta_al, 1e-2)
        grad_al = (p_curr - 0.0) / (theta_al * dx)
        
        # Apply using np.where to avoid chained indexing
        # Note: grad_p_x is already 0.0.
        # We update it where mask_u_inner is True
        
        # Combine valid conditions with inner mask
        # We apply safe updates.
        
        # Slicing: [:, 1:-1] has shape (ny, nx-2)
        # Conditions (is_LL etc) have shape (ny, nx).
        # We need to slice conditions to (ny, nx-2) to match val_x_slice.
        # u inner range: i from 1 to nx-1.
        # p_curr is p[j, i]. If i=1..nx-1, indices are 1..nx-1.
        # p matches slice [:, 1:-1].
        # So we slice conditions with [:, 1:]?
        # Let's check:
        # p_curr = p. Condition logic `is_LL = (phi_L < 0) & (phi_R < 0)`.
        # phi_L = phi_prev (rolled), phi_R = phi_curr (phi).
        # `is_LL` calculation is valid for whole array.
        # We need `is_LL` at indices where we update Grad P.
        # Grad P update idx: 1 to nx-1.
        # Corresponding `is_LL` is at `[:, 1:]`?
        # p_curr uses p[:, 1:]. p_prev uses p[:, 0:-1]?
        # Earlier I did: (p_curr[:, 1:] - p_prev[:, 1:])
        # So I used slice [:, 1:] relative to the `rolled` array.
        # Let's verify `p_curr[:, 1:]` corresponds to `p[:, 1:]`.
        # `p_prev` is rolled left (shift index i to i-1). `p_prev[:, 1]` holds old `p[:, 0]`.
        # So `p_prev[:, 1:]` holds old `p[:, :-1]`.
        # This matches `(p[i] - p[i-1])`.
        # So `is_LL` (which uses same roll) should be sliced with `[:, 1:]`.
        
        # Wait, shape error says: (50,50) (50,50) (50,49).
        # Inner u count: 51 columns. 1:-1 is 1..49 (49 columns).
        # Conditions (50,50) when sliced `[:, 1:]` is (50,49).
        # Wait, 50-1=49.
        # So `is_LL[:, 1:]` is (50,49).
        # `val_x_slice` is (50,49).
        # So why broadcast error?
        # Maybe I didn't slice `is_LL` in the previous code?
        # Yes, I forgot to slice `is_LL` in the previous edit for X gradient. I only did it for Y gradient in the last step.
        
        is_LL_s_x = is_LL[:, 1:]
        is_LA_s_x = is_LA[:, 1:]
        is_AL_s_x = is_AL[:, 1:]
        grad_standard_s_x = grad_standard[:, 1:]
        grad_la_s_x = grad_la[:, 1:]
        grad_al_s_x = grad_al[:, 1:]
        
        # We operate on the slice [:, 1:-1] which has shape (ny, nx-1)
        # val_x_slice will hold the computed gradients for the inner faces
        val_x_slice = np.zeros_like(grad_p_x[:, 1:-1])
        
        val_x_slice = np.where(is_LL_s_x, grad_standard_s_x, val_x_slice)
        val_x_slice = np.where(is_LA_s_x, grad_la_s_x, val_x_slice)
        val_x_slice = np.where(is_AL_s_x, grad_al_s_x, val_x_slice)
        
        grad_p_x[:, 1:-1] = val_x_slice

        # --- V Gradient ---
        p_curr = p # p[j, i] (Top)
        p_prev = np.roll(p, 1, axis=0) # p[j-1, i] (Bottom)
        
        phi_curr = phi
        phi_prev = np.roll(phi, 1, axis=0)
        
        # V-faces: v[1, i] to v[ny-1, i]
        
        phi_D = phi_prev
        phi_U = phi_curr
        
        is_LL = (phi_D < 0) & (phi_U < 0)
        is_LA = (phi_D < 0) & (phi_U > 0)
        is_AL = (phi_D > 0) & (phi_U < 0)
        
        grad_standard = (p_curr - p_prev) / dy
        
        theta_la = abs(phi_D) / (abs(phi_D) + abs(phi_U) + 1e-12)
        theta_la = np.maximum(theta_la, 1e-2)
        grad_la = (0.0 - p_prev) / (theta_la * dy)
        
        theta_al = abs(phi_U) / (abs(phi_U) + abs(phi_D) + 1e-12)
        theta_al = np.maximum(theta_al, 1e-2)
        grad_al = (p_curr - 0.0) / (theta_al * dy)
        
        # Slicing: [1:-1, :] has shape (ny-2, nx)
        val_y_slice = np.zeros_like(grad_p_y[1:-1, :])
        
        # Ensure masks are also sliced to match (ny-2, nx)
        # Conditions are (ny-1, nx).
        # Inner loop for V is j from 1 to ny-1.
        # p_curr is p[j, i] (Top)
        # p_prev is p[j-1, i] (Bottom)
        # But wait, p_curr, p_prev were formed by np.roll(p, 1, axis=0) which is full size (ny, nx).
        # The conditions is_LL etc are also (ny, nx).
        # So we need to slice the conditions to [1:-1, :] as well.
        
        is_LL_s = is_LL[1:, :]
        is_LA_s = is_LA[1:, :]
        is_AL_s = is_AL[1:, :]
        grad_standard_s = grad_standard[1:, :]
        grad_la_s = grad_la[1:, :]
        grad_al_s = grad_al[1:, :]
        
        # Wait, shape mismatch:
        # grad_p_y[1:-1, :] shape is (48, 50).
        # is_LL shape is (50, 50).
        # is_LL[1:, :] shape is (49, 50).
        # Something is wrong with loop bounds implied by slice.
        # V grid: j=0..ny.
        # Inner faces: j=1..ny-1. Total ny-1 faces.
        # Slice [1:-1, :] excludes 0 and -1 so it has ny-2 items? No.
        # Array size (ny+1, nx).
        # [1:-1, :] means index 1 to ny-1 (inclusive of start, exclusive of end).
        # Wait. Python slice [1:-1] excludes last element.
        # If shape is 51, [1:-1] is 1..49 (49 elements).
        # self.v shape is (ny+1, nx) = (51, 50).
        # self.v[1:-1, :] is 1..49 -> 49 elements.
        
        # Conditions based on p (50, 50).
        # p_curr = p. p_prev = roll(p, 1).
        # j-th V-face connects p[j-1, i] and p[j, i].
        # For j=1, connects p[0] and p[1].
        # For j=ny-1 (49), connects p[48] and p[49].
        # So we want indices 1 to ny-1 (49 faces).
        # p indices: 0..ny-1.
        
        # np.roll shifts down. p_prev[j] is old p[j-1].
        # So at index j, logic uses p[j] and p[j-1]. Correct.
        # We need slice j=1 to j=ny-1?
        # range is 1 to 49 (excluding 50).
        # In Python: slice [1:] gives 1..49 (49 items).
        
        # self.v has 51 items (0..50).
        # We need 1..49.
        # self.v[1:-1] is 1..49. Correct?
        # If size 51: 0, 1, ..., 49, 50.
        # [1:-1] is 1, ..., 49. Length 49.
        
        # Conditions (from p 50x50):
        # We need j=1 to j=49?
        # p has max index 49.
        # Relation V_j <-> P_j, P_{j-1}.
        # For j=50 (Top wall), P_50 doesn't exist. V_50 is BC.
        # So we process j=1 to j=ny-1 (49).
        
        # Conditions slicing:
        # is_LL has shape (50, 50).
        # At j=1, matches V_1.
        # At j=49, matches V_49.
        # So we need is_LL[1:, :]. Length 49.
        
        # But wait, earlier error said:
        # operands could not be broadcast together with shapes (50,50) (50,50) (50,49)
        # That was for X gradient? (50,51) vs (50,50).
        # No, error 2 was: (50,50) (50,50) (50,49).
        # The 49 is val_x_slice.
        # 50,50 is condition.
        # So condition needs to be sliced.
        
        val_y_slice = np.where(is_LL[1:, :], grad_standard[1:, :], val_y_slice)
        val_y_slice = np.where(is_LA[1:, :], grad_la[1:, :], val_y_slice)
        val_y_slice = np.where(is_AL[1:, :], grad_al[1:, :], val_y_slice)
        grad_p_y[1:-1, :] = val_y_slice
        
        return grad_p_x, grad_p_y

    def solve_pressure(self, u_star, v_star, iters=100):
        """Solve for pressure 'p' (Non-Incremental Projection)."""
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        dt = self.dt
        phi = self.phi

        # Compute RHS = div(u*) / dt
        div_u = (u_star[:, 1:] - u_star[:, :-1]) / dx + (v_star[1:, :] - v_star[:-1, :]) / dy
        rhs = div_u / dt

        # Initialize coefficients
        # Diagonals
        ap = np.zeros((ny, nx))
        # Neighbors
        ae = np.zeros((ny, nx))
        aw = np.zeros((ny, nx))
        an = np.zeros((ny, nx))
        as_ = np.zeros((ny, nx))
        
        rhs_bc = np.zeros((ny, nx)) # Zero bc for p (p=0 at interface)

        # --- Compute Coefficients ---
        mask_liquid = (phi < 0)
        
        inv_dx2 = 1.0 / dx**2
        inv_dy2 = 1.0 / dy**2
        
        ae[:, :] = inv_dx2
        aw[:, :] = inv_dx2
        an[:, :] = inv_dy2
        as_[:, :] = inv_dy2
        ap[:, :] = -2.0 * (inv_dx2 + inv_dy2)

        # Walls
        aw[:, 0] = 0.0
        ap[:, 0] += inv_dx2
        ae[:, -1] = 0.0
        ap[:, -1] += inv_dx2
        as_[0, :] = 0.0
        ap[0, :] += inv_dy2
        an[-1, :] = 0.0
        ap[-1, :] += inv_dy2
        
        # Free Surface (Cheng's BC)
        # Using p=0 at interface. Ghost value p_ghost = -p_internal / theta if linear?
        # Cheng BC form: ap * p_P = (neighbors) - ...
        # For Dirichlet p=0:
        # Standard Finite Difference near boundary:
        # (p_E - p_P)/ (theta * dx * dx) ...
        # Effectively, the coefficient becomes different.
        # My implementation matches the standard "Ghost Fluid Method for Poisson Equation" (Cheng & Armfield)
        # where we modify the stencil to account for the distance to p=0 constraint.
        
        # East
        phi_east = np.roll(phi, -1, axis=1)
        is_interface_east = (phi < 0) & (phi_east > 0)
        is_interface_east[:, -1] = False
        theta_e = abs(phi) / (abs(phi) + abs(phi_east) + 1e-12)
        
        mask = is_interface_east
        theta = np.maximum(theta_e[mask], 1e-2)
        ae[mask] = 0.0
        ap[mask] += inv_dx2
        ap[mask] -= 1.0 / (theta * dx**2)
        
        # West
        phi_west = np.roll(phi, 1, axis=1)
        is_interface_west = (phi < 0) & (phi_west > 0)
        is_interface_west[:, 0] = False
        theta_w = abs(phi) / (abs(phi) + abs(phi_west) + 1e-12)
        
        mask = is_interface_west
        theta = np.maximum(theta_w[mask], 1e-2)
        aw[mask] = 0.0
        ap[mask] += inv_dx2
        ap[mask] -= 1.0 / (theta * dx**2)

        # North
        phi_north = np.roll(phi, -1, axis=0)
        is_interface_north = (phi < 0) & (phi_north > 0)
        is_interface_north[-1, :] = False
        theta_n = abs(phi) / (abs(phi) + abs(phi_north) + 1e-12)
        
        mask = is_interface_north
        theta = np.maximum(theta_n[mask], 1e-2)
        an[mask] = 0.0
        ap[mask] += inv_dy2
        ap[mask] -= 1.0 / (theta * dy**2)
        
        # South
        phi_south = np.roll(phi, 1, axis=0)
        is_interface_south = (phi < 0) & (phi_south > 0)
        is_interface_south[0, :] = False
        theta_s = abs(phi) / (abs(phi) + abs(phi_south) + 1e-12)
        
        mask = is_interface_south
        theta = np.maximum(theta_s[mask], 1e-2)
        as_[mask] = 0.0
        ap[mask] += inv_dy2
        ap[mask] -= 1.0 / (theta * dy**2)
        
        # --- Solve (SOR) for Pressure (p) ---
        p = self.p.copy() # Warm start
        omega = 1.0 # Gauss-Seidel (Stable)
        
        total_rhs = rhs - rhs_bc
        
        for _ in range(iters):
            p_e = np.roll(p, -1, axis=1)
            p_w = np.roll(p, 1, axis=1)
            p_n = np.roll(p, -1, axis=0)
            p_s = np.roll(p, 1, axis=0)
            
            p_src = total_rhs - (ae * p_e + aw * p_w + an * p_n + as_ * p_s)
            p_new = p_src / ap
            
            p[mask_liquid] = (1.0 - omega) * p[mask_liquid] + omega * p_new[mask_liquid]
            p[~mask_liquid] = 0.0 # Force 0/Atmosphere in air
            
        self.p = p

    def correct_velocity(self, u_star, v_star):
        """Correct velocity using pressure gradient."""
        dt = self.dt
        
        # Compute gradient of pressure (p)
        # Using self.p which satisfies p=0 at interface
        grad_p_x, grad_p_y = self.compute_pressure_gradient(self.p, self.phi)
        
        self.u[:, 1:-1] = u_star[:, 1:-1] - dt * grad_p_x[:, 1:-1]
        self.v[1:-1, :] = v_star[1:-1, :] - dt * grad_p_y[1:-1, :]
        
    def _should_reinitialize(self) -> bool:
        """Determine if reinitialization is required based on the selected mode."""
        
        # 1. Simple Interval Method (Default)
        if self.reinit_mode == "interval":
            return self.step_count % self.reinit_interval == 0
            
        # 2. Adaptive Method (Gradient Distortion)
        elif self.reinit_mode == "adaptive":
            # Safety Net
            if self.step_count % self.reinit_interval == 0:
                print(f"  -> Triggering Reinitialization (Reason: Interval Safety)")
                return True
                
            # Check distortion: E = median(| |grad(phi)| - 1 |) near interface
            phi_val = self.phi
            dx, dy = self.dx, self.dy
            
            # Central difference for smoothness check
            phix = (np.roll(phi_val, -1, axis=1) - np.roll(phi_val, 1, axis=1)) / (2 * dx)
            phiy = (np.roll(phi_val, -1, axis=0) - np.roll(phi_val, 1, axis=0)) / (2 * dy)
            grad_norm = np.sqrt(phix**2 + phiy**2)
            
            bandwidth = 5.0 * max(dx, dy)
            mask_narrow = (np.abs(phi_val) < bandwidth)
            
            if np.any(mask_narrow):
                distortion_local = np.abs(grad_norm[mask_narrow] - 1.0)
                E_median = np.median(distortion_local)
                if self.step_count % 10 == 0:
                    print(f"Step {self.step_count}: Reinit Distortion E_median = {E_median:.4f}")
            else:
                E_median = 0.0
                
            E_threshold_median = 0.1
            
            if E_median > E_threshold_median:
                print(f"  -> Triggering Reinitialization (Reason: Distortion E={E_median:.4f})")
                return True
            
            return False
            
        return False

    def solve_ns_step(self):
        """Execute one full time step with performance profiling."""
        t_start_step = time.perf_counter()
        self.step_count += 1
        
        t_extrap = 0.0
        
        # 1. Extrapolate Velocity (Liquid -> Air)
        t0 = time.perf_counter()
        self.u, self.v = self.extrapolate_velocity()
        t_extrap += (time.perf_counter() - t0)
        
        # 2. Advect Level Set
        t0 = time.perf_counter()
        self.advect_levelset()
        t1 = time.perf_counter()
        self.perf_log["Level Set Advect"].append(t1 - t0)
        
        # 3. Reinitialize Level Set
        t0 = time.perf_counter()
        if self._should_reinitialize():
            self.reinitialize_levelset(iterations=10)
        t1 = time.perf_counter()
        self.perf_log["Level Set Reinit"].append(t1 - t0)
        
        # 4. NS Predictor
        t0 = time.perf_counter()
        u_star, v_star = self.compute_tentative_velocity()
        t1 = time.perf_counter()
        self.perf_log["Tentative Velocity"].append(t1 - t0)
        
        # 4.5 Extrapolate u_star, v_star (Liquid -> Air)
        t0 = time.perf_counter()
        u_star, v_star = self.extrapolate_velocity(u_star, v_star)
        t_extrap += (time.perf_counter() - t0)
        self.perf_log["Velocity Extrapolation"].append(t_extrap)
        
        # 5. Pressure Solve (Non-Incremental)
        t0 = time.perf_counter()
        self.solve_pressure(u_star, v_star)
        t1 = time.perf_counter()
        self.perf_log["Pressure Solve"].append(t1 - t0)
        
        # 6. Corrector
        t0 = time.perf_counter()
        self.correct_velocity(u_star, v_star)
        t1 = time.perf_counter()
        self.perf_log["Correction"].append(t1 - t0)

        t_end_step = time.perf_counter()
        self.perf_log["Total Step"].append(t_end_step - t_start_step)

    def plot_performance_stats(self, save_path="performance_profile.png"):
        """Plot the collected performance statistics."""
        if not self.perf_log["Total Step"]:
            print("No performance data collected.")
            return

        # Calculate averages and totals
        labels = []
        avgs = []
        totals = []
        
        for key, times in self.perf_log.items():
            if key == "Total Step": continue
            labels.append(key)
            if len(times) > 0:
                avgs.append(sum(times) / len(times))
                totals.append(sum(times))
            else:
                avgs.append(0)
                totals.append(0)
        
        # Pie Chart of Total Time Distribution
        try:
            fig, ax = plt.subplots(1, 2, figsize=(14, 6))
            
            ax[0].pie(totals, labels=labels, autopct='%1.1f%%', startangle=140)
            ax[0].set_title("Total Computation Time Distribution")
            
            # Bar Chart of Average Time per Step
            y_pos = np.arange(len(labels))
            ax[1].barh(y_pos, avgs, align='center')
            ax[1].set_yticks(y_pos)
            ax[1].set_yticklabels(labels)
            ax[1].invert_yaxis()  # labels read top-to-bottom
            ax[1].set_xlabel('Average Time (s)')
            ax[1].set_title('Average Execution Time per Step')
            
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            print(f"Performance profile saved to {save_path}")
        except Exception as e:
            print(f"Failed to plot performance stats: {e}")


if __name__ == "__main__":
    # Parameters
    nx, ny = 50, 50
    re = 1000.0
    dt = 0.001
    # Change initial_condition to "column" for dam break, "flat" for cup of water
    solver = LevelSetFlowSolver(nx=nx, ny=ny, re=re, dt=dt, initial_condition="column")
    
    steps = 3000
    print(f"Starting Level Set Simulation for {steps} steps...")
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    
    for i in range(steps):
        solver.solve_ns_step()
        
        if (i + 1) % 50 == 0:
            print(f"Step {i+1}/{steps}")
            ax.clear()
            
            # Plot Level Set (phi=0 is interface)
            # Water (phi < 0) is Blue
            contour = ax.contourf(solver.X, solver.Y, solver.phi, levels=[-10, 0, 10], colors=['blue', 'cyan'], alpha=0.5)
            ax.contour(solver.X, solver.Y, solver.phi, levels=[0], colors='black', linewidths=2)
            
            # Plot Velocity Quiver (Subsampled)
            # u_c, v_c = solver._get_advection_velocity()
            # skip = 2
            # ax.quiver(solver.X[::skip, ::skip], solver.Y[::skip, ::skip], 
            #           u_c[::skip, ::skip], v_c[::skip, ::skip], scale=5)
            
            # Title with mass (area of water)
            water_area = np.sum(solver.phi < 0) * solver.dx * solver.dy
            ax.set_title(f"Step {i+1}, Water Area={water_area:.4f}")
            
            plt.pause(0.01)
            
    plt.ioff()
    plt.show()
