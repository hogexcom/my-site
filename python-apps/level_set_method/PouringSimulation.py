
import numpy as np
import matplotlib.pyplot as plt
import os
from LevelSetFlowSolver import LevelSetFlowSolver
from ParticleLevelSetSolver import ParticleLevelSetSolver

# Configuration
USE_PLS = True

def run_pouring_simulation():
    # 1. Setup Solver
    nx, ny = 100, 100
    
    if USE_PLS:
        print("Using Particle Level Set Solver")
        solver = ParticleLevelSetSolver(
            nx=nx, ny=ny, 
            lx=1.0, ly=1.0, 
            re=1000.0, 
            dt=0.001, 
            initial_condition="empty",
            reinit_mode="interval"
        )
    else:
        print("Using Standard Level Set Solver")
        solver = LevelSetFlowSolver(
            nx=nx, ny=ny, 
            lx=1.0, ly=1.0, 
            re=1000.0, 
            dt=0.001, 
            initial_condition="empty",
            reinit_mode="interval"
        )
        
    solver.reinit_interval = 10 # Reduce reinit frequency to save mass
    
    # 3. Define Source (Faucet) Parameters
    # Pre-fill bottom with water (y < 0.1) as requested
    solver.phi = np.minimum(solver.phi, solver.Y - 0.1)
    
    # Re-initialize particles for PLS because we manually changed phi to add bottom water
    if isinstance(solver, ParticleLevelSetSolver):
        print("Re-seeding particles for Pouring initial condition (with bottom layer)...")
        solver.init_particles()

    faucet_x_min = 0.45
    faucet_x_max = 0.55 # Widened to 0.15
    faucet_y_min = 0.90
    faucet_y_max = 1.00 # Top of domain
    faucet_vel = -1.0   # Downward velocity
    
    # Helper to enforce source
    def apply_source(solver):
        X, Y = solver.X, solver.Y
        
        # Geometry: Rectangle for simplicity
        # Signed distance to rectangle [xmin, xmax] x [ymin, ymax]
        # max(x - xmax, xmin - x, y - ymax, ymin - y)
        d_box = np.maximum.reduce([
            X - faucet_x_max,
            faucet_x_min - X,
            Y - faucet_y_max,
            faucet_y_min - Y
        ])
        
        # Enforce water inside box (phi < 0)
        # phi_new = min(phi_old, d_box)
        solver.phi = np.minimum(solver.phi, d_box)
        
        # Enforce Velocity
        # Staggered v is at (ny+1, nx)
        # Locations:
        # We need to find v-cells that fall inside the source region
        dy = solver.dy
        dx = solver.dx
        y_v = np.linspace(0, solver.lx, ny + 1) # Fix ly? solver.ly
        y_v = np.linspace(0, solver.ly, ny + 1)
        x_v = np.linspace(dx/2, solver.lx - dx/2, nx)
        X_v, Y_v = np.meshgrid(x_v, y_v)
        
        # Mask for v
        mask_v_source = (X_v >= faucet_x_min) & (X_v <= faucet_x_max) & \
                        (Y_v >= faucet_y_min)
        
        # Apply velocity v
        solver.v[mask_v_source] = faucet_vel
        
        # Mask for u (force zero horizontal velocity in source)
        x_u = np.linspace(0, solver.lx, nx + 1)
        y_u = np.linspace(dy/2, solver.ly - dy/2, ny)
        X_u, Y_u = np.meshgrid(x_u, y_u)
        
        mask_u_source = (X_u >= faucet_x_min) & (X_u <= faucet_x_max) & \
                        (Y_u >= faucet_y_min)
        
        solver.u[mask_u_source] = 0.0
        
    # 4. Run Simulation
    T_end = 2.0 # Run longer to fill
    steps = int(T_end / solver.dt)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "pouring_output")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting Pouring Simulation: T=0 to {T_end}, steps={steps}")
    
    # Prepare interactive plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(6,6))

    for n in range(steps):
        # Apply inflow BC at the beginning of step
        apply_source(solver)
        
        solver.solve_ns_step()
        
        t = (n + 1) * solver.dt
        if (n + 1) % 50 == 0:
            u_bottom_max = np.max(np.abs(solver.u[0, :]))
            phi_min_bottom = np.min(solver.phi[0, :])
            # Count pixels with phi < 0 at bottom
            wetted_count = np.sum(solver.phi[0, :] < 0)
            wetted_width = wetted_count * solver.dx
            print(f"Step {n+1}/{steps}, t={t:.3f}, Max |u_bottom|={u_bottom_max:.4f}, Min phi_bottom={phi_min_bottom:.4f}, Width={wetted_width:.3f}")
            plot_frame(solver, n + 1, t, output_dir, ax)
            
    # Save performance profile
    solver.plot_performance_stats(save_path=f"{output_dir}/performance_profile.png")
    plt.ioff()
    plt.show()

def plot_frame(solver, step, t, output_dir, ax):
    ax.clear()
    
    # Plot Level Set Interface (phi=0)
    # Using cell centers for contour
    if np.any(solver.phi < 0):
        ax.contour(solver.X, solver.Y, solver.phi, levels=[0], colors='blue', linewidths=2)
        # Fill water
        ax.contourf(solver.X, solver.Y, solver.phi, levels=[-10, 0], colors=['#a0c0ff'], alpha=0.5)
    
    # Plot Particles if PLS
    if isinstance(solver, ParticleLevelSetSolver):
        if len(solver.neg_particles_x) > 0:
            ax.scatter(solver.neg_particles_x, solver.neg_particles_y, s=1, c='red', alpha=0.1)  # Water
        if len(solver.pos_particles_x) > 0:
            ax.scatter(solver.pos_particles_x, solver.pos_particles_y, s=1, c='green', alpha=0.1) # Air

    ax.set_title(f"Pouring Simulation t={t:.3f}")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_aspect('equal')
    
    # Draw Cup walls (Visual only, domain boundary is the wall)
    ax.plot([0, 0, 1, 1], [1, 0, 0, 1], 'k-', linewidth=3)
    
    filename = f"{output_dir}/pour_{step:04d}.png"
    plt.savefig(filename)
    plt.draw()
    plt.pause(0.05)

if __name__ == "__main__":
    run_pouring_simulation()
