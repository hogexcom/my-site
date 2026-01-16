import numpy as np
import matplotlib.pyplot as plt
import time
from LevelSetFlowSolver import LevelSetFlowSolver

class ParticleLevelSetSolver(LevelSetFlowSolver):
    def __init__(self, nx=50, ny=50, re=1000.0, dt=0.001, lx=1.0, ly=1.0, 
                 initial_condition="flat", reinit_mode="interval", particles_per_cell=64):
        super().__init__(nx, ny, re, dt, lx, ly, initial_condition, reinit_mode)
        
        # PLS Parameters
        self.particles_per_cell = particles_per_cell
        self.particle_band = 3 * max(self.dx, self.dy) # Seed within 3 cells
        self.rmin = 0.1 * self.dx
        self.rmax = 0.5 * self.dx
        
        # Particle Arrays (Struct of Arrays)
        self.pos_particles_x = np.array([])
        self.pos_particles_y = np.array([])
        self.pos_particles_r = np.array([])
        
        self.neg_particles_x = np.array([])
        self.neg_particles_y = np.array([])
        self.neg_particles_r = np.array([])
        
        # Initialize Particles
        self.init_particles()
        
    def init_particles(self):
        """Seed particles near the interface."""
        print("Initializing Particles with improved logic...")
        
        p_pos_x, p_pos_y, p_pos_r = [], [], []
        p_neg_x, p_neg_y, p_neg_r = [], [], []
        
        dx, dy = self.dx, self.dy
        
        for j in range(self.ny):
            for i in range(self.nx):
                # Check neighbors to see if interface is near (efficient band check)
                # Simply: if |phi| < band, seed.
                phi_c = self.phi[j, i]
                
                if abs(phi_c) < self.particle_band:
                    for _ in range(self.particles_per_cell):
                        # Random pos
                        px = (i + np.random.rand()) * dx
                        py = (j + np.random.rand()) * dy
                        
                        # Interpolate phi at (px, py) for accurate sign
                        phi_p = self.get_phi_at(px, py)
                        
                        # Radius based on phi_p
                        radius = np.clip(abs(phi_p), self.rmin, self.rmax)
                        
                        if phi_p > 0:
                            p_pos_x.append(px); p_pos_y.append(py); p_pos_r.append(radius)
                        else:
                            p_neg_x.append(px); p_neg_y.append(py); p_neg_r.append(radius)
                            
        self.pos_particles_x = np.array(p_pos_x)
        self.pos_particles_y = np.array(p_pos_y)
        self.pos_particles_r = np.array(p_pos_r)
        
        self.neg_particles_x = np.array(p_neg_x)
        self.neg_particles_y = np.array(p_neg_y)
        self.neg_particles_r = np.array(p_neg_r)
        
        print(f"Initialized Pos: {len(self.pos_particles_x)}, Neg: {len(self.neg_particles_x)}")

    def get_phi_at(self, px, py):
        """Bilinear interpolation of Level Set phi."""
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        
        x_grid = (px / dx) - 0.5
        y_grid = (py / dy) - 0.5
        
        i = np.floor(x_grid).astype(int); j = np.floor(y_grid).astype(int)
        i = np.clip(i, 0, nx - 2); j = np.clip(j, 0, ny - 2)
        
        wx = x_grid - i; wy = y_grid - j
        
        phi00 = self.phi[j, i]; phi10 = self.phi[j, i+1]
        phi01 = self.phi[j+1, i]; phi11 = self.phi[j+1, i+1]
        
        return (1-wx)*(1-wy)*phi00 + wx*(1-wy)*phi10 + (1-wx)*wy*phi01 + wx*wy*phi11

    def solve_ns_step(self):
        """
        Modified Time Stepping for PLS.
        1. Advect Grid Level Set & Particles
        2. Correct Level Set (Error Correction)
        3. Reinitialize Level Set
        4. Standard NS (Pressure, Velocity)
        5. Reseed
        """
        # Timing
        t_start = time.perf_counter()
        
        # 1. Extrapolate Velocity (Liquid -> Air) for stability
        self.u, self.v = self.extrapolate_velocity()
        
        # 2. Advect Grid Level Set
        self.advect_levelset()
        
        # 3. Advect Particles (Use same velocity field)
        self.advect_particles()
        
        # 4. Correct Level Set (Using Escaped Particles)
        self.correct_levelset()

        # 5. Reinitialize (Smooths the corrected field)
        # Always reinit for PLS to propagate correction
        # This also helps remove local minima created by stray particles before cleanup
        self.reinitialize_levelset(iterations=1) # PLS requires more reinit iterations
        
        # 5.5 Reproject escaped particles back to the interface
        self.reproject_escaped_particles()

        # 6. Reseed (Every 10 or 20 steps)
        # cleanup uses phi. If we reinit first, stray particles formed far from interface
        # will see 'smoothed' phi (large dist) and be deleted.
        if self.step_count % 5 == 0:
            self.reseed_particles()
        
        # 7. NS Predictor
        u_star, v_star = self.compute_tentative_velocity()
        u_star, v_star = self.extrapolate_velocity(u_star, v_star)
        
        # 8. Pressure Solve
        self.solve_pressure(u_star, v_star)
        
        # 9. Corrector
        self.correct_velocity(u_star, v_star)
            
        self.step_count += 1
        
        # Logging
        if hasattr(self, 'perf_log'):
            self.perf_log["Total Step"].append(time.perf_counter() - t_start)


    def advect_particles(self):
        """Advect particles using RK2 and Staggered Grid Interpolation."""
        dt = self.dt
        
        def advect(px, py):
            # RK2 Midpoint
            u1, v1 = self.get_velocity_at(px, py)
            x_mid = px + 0.5 * dt * u1
            y_mid = py + 0.5 * dt * v1
            
            # Clamp Mid
            x_mid = np.clip(x_mid, 0, self.lx)
            y_mid = np.clip(y_mid, 0, self.ly)
            
            u2, v2 = self.get_velocity_at(x_mid, y_mid)
            px_new = px + dt * u2
            py_new = py + dt * v2
            
            # Boundary Clamp
            px_new = np.clip(px_new, 0, self.lx)
            py_new = np.clip(py_new, 0, self.ly)
            return px_new, py_new

        self.pos_particles_x, self.pos_particles_y = advect(self.pos_particles_x, self.pos_particles_y)
        self.neg_particles_x, self.neg_particles_y = advect(self.neg_particles_x, self.neg_particles_y)

    def get_velocity_at(self, px, py):
        """Interpolate Staggered Velocity (u, v) directly to particles."""
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        
        # --- Interpolate U ---
        x_grid_u = px / dx
        y_grid_u = (py / dy) - 0.5
        
        i_u = np.floor(x_grid_u).astype(int)
        j_u = np.floor(y_grid_u).astype(int)
        
        i_u = np.clip(i_u, 0, nx - 1)
        j_u = np.clip(j_u, 0, ny - 2)
        
        wx_u = x_grid_u - i_u
        wy_u = y_grid_u - j_u
        
        u00 = self.u[j_u, i_u]; u10 = self.u[j_u, i_u+1]
        u01 = self.u[j_u+1, i_u]; u11 = self.u[j_u+1, i_u+1]
        
        u_interp = (1-wx_u)*(1-wy_u)*u00 + wx_u*(1-wy_u)*u10 + (1-wx_u)*wy_u*u01 + wx_u*wy_u*u11
        
        # --- Interpolate V ---
        x_grid_v = (px / dx) - 0.5
        y_grid_v = py / dy
        
        i_v = np.floor(x_grid_v).astype(int)
        j_v = np.floor(y_grid_v).astype(int)
        
        i_v = np.clip(i_v, 0, nx - 2)
        j_v = np.clip(j_v, 0, ny - 1)
        
        wx_v = x_grid_v - i_v
        wy_v = y_grid_v - j_v
        
        v00 = self.v[j_v, i_v]; v10 = self.v[j_v, i_v+1]
        v01 = self.v[j_v+1, i_v]; v11 = self.v[j_v+1, i_v+1]
        
        v_interp = (1-wx_v)*(1-wy_v)*v00 + wx_v*(1-wy_v)*v10 + (1-wx_v)*wy_v*v01 + wx_v*wy_v*v11
        
        return u_interp, v_interp

    def correct_levelset(self):
        """
        Correct phi using ONLY escaped particles.
        Follows Enright et al. (2002):
        1. Construct phi_minus from escaped negative particles.
        2. Construct phi_plus from escaped positive particles.
        3. Merge: If collision, choose one with smaller magnitude (closest to interface).
        """
        dx, dy = self.dx, self.dy
        nx, ny = self.nx, self.ny
        
        # Initialize correction fields
        # phi_minus: Stores min(dist - r) for negative particles. Init with +inf.
        # phi_plus: Stores max(r - dist) aka -min(dist - r) for pos particles. Init with -inf.
        
        phi_minus_grid = np.full((ny, nx), np.inf)
        phi_plus_grid = np.full((ny, nx), -np.inf)
        
        # --- Helper to processing particles and update temp grids ---
        def process_particles(px, py, pr, p_type, target_grid):
            if len(px) == 0: return None
            
            # 1. Update Radius & Check Escape
            x_g = (px / dx) - 0.5
            y_g = (py / dy) - 0.5
            
            ig = np.floor(x_g).astype(int); jg = np.floor(y_g).astype(int)
            ig = np.clip(ig, 0, nx - 2); jg = np.clip(jg, 0, ny - 2)
            
            w_x = x_g - ig; w_y = y_g - jg
            
            p00 = self.phi[jg, ig]; p10 = self.phi[jg, ig+1]
            p01 = self.phi[jg+1, ig]; p11 = self.phi[jg+1, ig+1]
            
            phi_interp = (1-w_x)*(1-w_y)*p00 + w_x*(1-w_y)*p10 + (1-w_x)*w_y*p01 + w_x*w_y*p11
            
            # Update Radius
            new_r = np.clip(np.abs(phi_interp), self.rmin, self.rmax)
            
            # Check Escape
            if p_type == "negative":
                # Water: Escaped if phi > 0
                is_escaped = (phi_interp > 0)
            else:
                # Air: Escaped if phi < 0
                is_escaped = (phi_interp < 0)
                
            px_esk = px[is_escaped]
            py_esk = py[is_escaped]
            pr_esk = new_r[is_escaped]
            
            
            if len(px_esk) == 0: return new_r
            
            # 2. Apply to Grid Stencil
            offsets = np.array([-1, 0, 1])
            ox, oy = np.meshgrid(offsets, offsets)
            ox = ox.flatten(); oy = oy.flatten()
            
            i_c = np.round(px_esk / dx - 0.5).astype(int)
            j_c = np.round(py_esk / dy - 0.5).astype(int)
            
            IX = i_c[:, None] + ox[None, :]
            IY = j_c[:, None] + oy[None, :]
            
            valid = (IX >= 0) & (IX < nx) & (IY >= 0) & (IY < ny)
            
            ix_flat = IX[valid]
            iy_flat = IY[valid]
            
            px_flat = np.repeat(px_esk, 9).reshape(len(px_esk), 9)[valid]
            py_flat = np.repeat(py_esk, 9).reshape(len(px_esk), 9)[valid]
            pr_flat = np.repeat(pr_esk, 9).reshape(len(px_esk), 9)[valid]
            
            node_x = (ix_flat + 0.5) * dx
            node_y = (iy_flat + 0.5) * dy
            dist = np.sqrt((node_x - px_flat)**2 + (node_y - py_flat)**2)
            
            if p_type == "negative":
                # Water candidates: phi_p = dist - r (which is negative inside)
                val = dist - pr_flat
                np.minimum.at(target_grid, (iy_flat, ix_flat), val)
            else:
                # Air candidates: phi_p = r - dist (which is positive inside) (Enright: phi = max(phi, r-dist))
                # Store r - dist
                val = pr_flat - dist
                np.maximum.at(target_grid, (iy_flat, ix_flat), val)
                
            return new_r

        # --- Process ---
        new_r_neg = process_particles(self.neg_particles_x, self.neg_particles_y, self.neg_particles_r, "negative", phi_minus_grid)
        if new_r_neg is not None: self.neg_particles_r = new_r_neg
        
        new_r_pos = process_particles(self.pos_particles_x, self.pos_particles_y, self.pos_particles_r, "positive", phi_plus_grid)
        if new_r_pos is not None: self.pos_particles_r = new_r_pos
        
        # --- Merge Logic ---
        # 1. Negative Correction Candidates (phi_minus_grid < inf)
        # 2. Positive Correction Candidates (phi_plus_grid > -inf)
        
        has_neg = (phi_minus_grid < 1e10) 
        has_pos = (phi_plus_grid > -1e10)
        
        # Case A: Only Negative Correction
        mask_only_neg = has_neg & (~has_pos)
        self.phi[mask_only_neg] = np.minimum(self.phi[mask_only_neg], phi_minus_grid[mask_only_neg])
        
        # Case B: Only Positive Correction
        mask_only_pos = has_pos & (~has_neg)
        self.phi[mask_only_pos] = np.maximum(self.phi[mask_only_pos], phi_plus_grid[mask_only_pos])
        
        # Case C: Both Corrections (Collision) (Collision: both particles escaped and landed on same cell)
        mask_both = has_neg & has_pos
        if np.any(mask_both):
            # Compare magnitudes (dist to interface)
            mag_neg = np.abs(phi_minus_grid[mask_both])
            mag_pos = np.abs(phi_plus_grid[mask_both])
            
            p_curr = self.phi[mask_both]
            p_neg = phi_minus_grid[mask_both]
            p_pos = phi_plus_grid[mask_both]
            
            res_neg = np.minimum(p_curr, p_neg)
            res_pos = np.maximum(p_curr, p_pos)
            
            # Enright: use particle that is closest to interface (smallest magnitude phi)
            # This is because the one closer to interface is likely the 'dominant' error or valid surface.
            final_vals = np.where(mag_neg < mag_pos, res_neg, res_pos)
            
            self.phi[mask_both] = final_vals

        self.phi = np.clip(self.phi, -self.phi_cap, self.phi_cap)

    def reproject_escaped_particles(self):
        """Reproject escaped particles back to the interface along phi normals."""
        dx, dy = self.dx, self.dy
        nx, ny = self.nx, self.ny

        # Precompute grad(phi) on grid
        phix = np.zeros_like(self.phi)
        phiy = np.zeros_like(self.phi)

        phix[:, 1:-1] = (self.phi[:, 2:] - self.phi[:, :-2]) / (2.0 * dx)
        phix[:, 0] = (self.phi[:, 1] - self.phi[:, 0]) / dx
        phix[:, -1] = (self.phi[:, -1] - self.phi[:, -2]) / dx

        phiy[1:-1, :] = (self.phi[2:, :] - self.phi[:-2, :]) / (2.0 * dy)
        phiy[0, :] = (self.phi[1, :] - self.phi[0, :]) / dy
        phiy[-1, :] = (self.phi[-1, :] - self.phi[-2, :]) / dy

        def interp_field(field, px, py):
            if len(px) == 0:
                return np.array([])
            x_g = (px / dx) - 0.5
            y_g = (py / dy) - 0.5
            i = np.floor(x_g).astype(int)
            j = np.floor(y_g).astype(int)
            i = np.clip(i, 0, nx - 2)
            j = np.clip(j, 0, ny - 2)
            wx = x_g - i
            wy = y_g - j
            f00 = field[j, i]
            f10 = field[j, i + 1]
            f01 = field[j + 1, i]
            f11 = field[j + 1, i + 1]
            return (1 - wx) * (1 - wy) * f00 + wx * (1 - wy) * f10 + (1 - wx) * wy * f01 + wx * wy * f11

        def reproject(px, py, pr, sign):
            if len(px) == 0:
                return px, py
            phi_p = interp_field(self.phi, px, py)
            escaped = (phi_p < 0) if sign > 0 else (phi_p > 0)
            if not np.any(escaped):
                return px, py
            gx = interp_field(phix, px, py)
            gy = interp_field(phiy, px, py)
            norm = np.sqrt(gx * gx + gy * gy) + 1e-12
            nx_p = gx / norm
            ny_p = gy / norm
            shift = (phi_p - sign * pr)
            px_new = px.copy()
            py_new = py.copy()
            px_new[escaped] = px_new[escaped] - shift[escaped] * nx_p[escaped]
            py_new[escaped] = py_new[escaped] - shift[escaped] * ny_p[escaped]
            px_new = np.clip(px_new, 0.0, self.lx)
            py_new = np.clip(py_new, 0.0, self.ly)
            return px_new, py_new

        self.pos_particles_x, self.pos_particles_y = reproject(
            self.pos_particles_x, self.pos_particles_y, self.pos_particles_r, 1.0
        )
        self.neg_particles_x, self.neg_particles_y = reproject(
            self.neg_particles_x, self.neg_particles_y, self.neg_particles_r, -1.0
        )

    def reseed_particles(self):
        """Reseed particles in interface cells to maintain density."""
        dx, dy = self.dx, self.dy
        
        # 1. Delete far particles (|phi| > band)
        def cleanup(px, py, pr, p_type):
            if len(px) == 0: return px, py, pr
            
            # Get phi at particles (Nearest neighbor for speed in cleanup)
            x_g = (px / dx) - 0.5
            y_g = (py / dy) - 0.5
            i = np.clip(np.floor(x_g).astype(int), 0, self.nx-1)
            j = np.clip(np.floor(y_g).astype(int), 0, self.ny-1)
            
            p_phi = self.phi[j, i]
            
            # Strict Cleanup: Only keep particles within the band.
            # Even escaped particles should be removed if they drift too far,
            # as they are unlikely to contribute meaningfully and waste memory.
            keep = (np.abs(p_phi) < self.particle_band)
            
                
            return px[keep], py[keep], pr[keep]
            
        self.neg_particles_x, self.neg_particles_y, self.neg_particles_r = cleanup(self.neg_particles_x, self.neg_particles_y, self.neg_particles_r, "negative")
        self.pos_particles_x, self.pos_particles_y, self.pos_particles_r = cleanup(self.pos_particles_x, self.pos_particles_y, self.pos_particles_r, "positive")
        
        # 2. Active Injection
        interf_mask = (np.abs(self.phi) < self.particle_band)
        
        # Histogram counting
        x_edges = np.arange(self.nx + 1) * dx
        y_edges = np.arange(self.ny + 1) * dy
        
        counts_neg, _, _ = np.histogram2d(self.neg_particles_x, self.neg_particles_y, bins=[x_edges, y_edges])
        counts_pos, _, _ = np.histogram2d(self.pos_particles_x, self.pos_particles_y, bins=[x_edges, y_edges]) # Returns shape (nx, ny)
        
        # histogram2d returns (nx, ny) where x corresponds to first bin array.
        # But our grid is self.phi[j, i] -> (y, x).
        # So counts[i, j] matches self.phi[j, i] if we are careful.
        # counts_neg[i, j] is count in cell x[i]..x[i+1], y[j]..y[j+1].
        # So we need to Transpose to align with (y, x) indexing of self.phi?
        # Yes. Verify: i is x-index, j is y-index.
        # self.phi[j, i].
        
        counts_neg = counts_neg.T # Now (ny, nx)
        counts_pos = counts_pos.T # Now (ny, nx)
        
        target = self.particles_per_cell // 2 
        
        # Negative Seeding (Water side: phi < 0, or just generally under-populated in band)
        # Note: Enright says "Seed both types in respective regions".
        # Neg particles in phi < 0. Pos particles in phi > 0.
        neg_seed_mask = (self.phi < 0) & interf_mask & (counts_neg < target)
        pos_seed_mask = (self.phi > 0) & interf_mask & (counts_pos < target)
        
        def inject(mask, p_type):
            rows, cols = np.where(mask)
            if len(rows) == 0: return np.array([]), np.array([]), np.array([])
            
            n_cells = len(rows)
            n_inj = 2 # Add fewer particles to be gentle
            total_new = n_cells * n_inj
            
            cx = (cols + 0.5) * dx
            cy = (rows + 0.5) * dy
            
            cx_rep = np.repeat(cx, n_inj)
            cy_rep = np.repeat(cy, n_inj)
            
            jitter_x = (np.random.rand(total_new) - 0.5) * dx
            jitter_y = (np.random.rand(total_new) - 0.5) * dy
            
            vals_x = cx_rep + jitter_x
            vals_y = cy_rep + jitter_y
            vals_r = np.full(total_new, self.rmin) # Safe init
            
            return vals_x, vals_y, vals_r
            
        new_nx, new_ny, new_nr = inject(neg_seed_mask, "negative")
        new_px, new_py, new_pr = inject(pos_seed_mask, "positive")
        
        if len(new_nx) > 0:
            if len(self.neg_particles_x) < 25000: # Global Cap
                self.neg_particles_x = np.concatenate([self.neg_particles_x, new_nx])
                self.neg_particles_y = np.concatenate([self.neg_particles_y, new_ny])
                self.neg_particles_r = np.concatenate([self.neg_particles_r, new_nr])
            
        if len(new_px) > 0:
            if len(self.pos_particles_x) < 25000: # Global Cap
                self.pos_particles_x = np.concatenate([self.pos_particles_x, new_px])
                self.pos_particles_y = np.concatenate([self.pos_particles_y, new_py])
                self.pos_particles_r = np.concatenate([self.pos_particles_r, new_pr])


if __name__ == "__main__":
    # Parameters
    nx, ny = 50, 50
    re = 1000.0
    dt = 0.001
    
    # Example Run
    solver = ParticleLevelSetSolver(nx=nx, ny=ny, re=re, dt=dt, initial_condition="column")
    
    steps = 3000
    print(f"Starting Level Set Simulation for {steps} steps...")
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    
    for i in range(steps):
        solver.solve_ns_step()
        
        if (i + 1) % 50 == 0:
            print(f"Step {i+1}/{steps} | P_Pos: {len(solver.pos_particles_x)}, P_Neg: {len(solver.neg_particles_x)}")
            ax.clear()
            
            # Plot Level Set
            contour = ax.contourf(solver.X, solver.Y, solver.phi, levels=[-10, 0, 10], colors=['blue', 'cyan'], alpha=0.5)
            ax.contour(solver.X, solver.Y, solver.phi, levels=[0], colors='black', linewidths=2)
            
            # Plot Particles
            if len(solver.neg_particles_x) > 0:
                ax.scatter(solver.neg_particles_x, solver.neg_particles_y, s=1, c='red', alpha=0.1, label='Neg (Water)')
            
            if len(solver.pos_particles_x) > 0:
                ax.scatter(solver.pos_particles_x, solver.pos_particles_y, s=1, c='green', alpha=0.1, label='Pos (Air)')
            
            water_area = np.sum(solver.phi < 0) * solver.dx * solver.dy
            ax.set_title(f"Step {i+1}, Area={water_area:.4f}, N_p={len(solver.pos_particles_x)+len(solver.neg_particles_x)}")
            
            plt.pause(0.01)
            
    plt.ioff()
    plt.show()
