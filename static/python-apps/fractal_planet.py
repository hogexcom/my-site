"""
Fractal Planet Generator using Random Great Circles (Fault Lines)

This program generates fractal terrain on a spherical planet by:
1. Randomly generating great circles (planes through Earth's center)
2. Treating each great circle as a fault line
3. Raising one side and lowering the other side
4. Repeating this process to create fractal-like terrain
5. Visualizing in 2D (Plate Carrée) and 3D (textured sphere)
"""

import numpy as np
import io
import base64


def _compute_side_jit(lon_grid, lat_grid, n):
    """
    JIT-compiled function to compute which side of great circle each point is on.

    Parameters
    ----------
    lon_grid : ndarray
        Longitude grid
    lat_grid : ndarray
        Latitude grid
    n : ndarray
        Normal vector (n_x, n_y, n_z)

    Returns
    -------
    side : ndarray
        Sign of dot product (+1, -1, or 0)
    """
    # Convert lat/lon to 3D Cartesian coordinates on unit sphere
    x = np.cos(lat_grid) * np.cos(lon_grid)
    y = np.cos(lat_grid) * np.sin(lon_grid)
    z = np.sin(lat_grid)

    # Compute dot product n·r
    dot_product = n[0] * x + n[1] * y + n[2] * z

    # Determine side
    return np.sign(dot_product)


class FractalPlanet:
    def __init__(self, width=800, height=400):
        """
        Initialize fractal planet generator.

        Parameters
        ----------
        width : int
            Width of the map in pixels (longitude resolution)
        height : int
            Height of the map in pixels (latitude resolution)
        """
        self.width = width
        self.height = height

        # Create coordinate grids for Plate Carrée projection
        # λ (longitude): -π to π
        # φ (latitude): -π/2 to π/2
        self.lon = np.linspace(-np.pi, np.pi, width)
        self.lat = np.linspace(-np.pi / 2, np.pi / 2, height)
        self.lon_grid, self.lat_grid = np.meshgrid(self.lon, self.lat)

        # Initialize height map (all zeros initially)
        self.height_map = np.zeros((height, width))

    def random_normal_vector(self):
        """
        Generate a random unit normal vector for a plane through origin.

        Returns
        -------
        n : ndarray
            Unit normal vector (n_x, n_y, n_z)
        """
        # Generate random point on unit sphere using normal distribution
        n = np.random.randn(3)
        # Normalize to unit vector
        n = n / np.linalg.norm(n)
        return n

    def compute_great_circle_side(self, n):
        """
        Compute which side of the great circle each point is on.

        For a plane with normal n = (n_x, n_y, n_z) through origin,
        a point r = (x, y, z) on the sphere is on the positive side if n·r > 0.

        Parameters
        ----------
        n : ndarray
            Unit normal vector (n_x, n_y, n_z)

        Returns
        -------
        side : ndarray
            Array of same shape as height_map, with values:
            +1 for positive side
            -1 for negative side
            0 for points on the circle (rare)
        """
        return _compute_side_jit(self.lon_grid, self.lat_grid, n)

    def add_fault(self, displacement=0.1):
        """
        Add a random fault line (great circle) to the terrain.

        Parameters
        ----------
        displacement : float
            Amount to displace the terrain on each side
        """
        # Generate random fault plane
        n = self.random_normal_vector()

        # Determine which side of the fault each point is on
        side = self.compute_great_circle_side(n)

        # Randomly decide which side goes up and which goes down
        if np.random.rand() > 0.5:
            side = -side

        # Apply displacement
        self.height_map += side * displacement

    def generate_terrain(self, num_faults=100, initial_displacement=1.0, decay=0.9):
        """
        Generate fractal terrain by applying multiple faults.

        Parameters
        ----------
        num_faults : int
            Number of fault lines to apply
        initial_displacement : float
            Initial displacement amount
        decay : float
            Decay factor for displacement (each fault has displacement *= decay)
        """
        displacement = initial_displacement

        for i in range(num_faults):
            self.add_fault(displacement)
            displacement *= decay

    def get_texture_array(self):
        """
        Convert height map to RGBA texture array for 3D visualization.
        Uses Earth-like colormap.

        Returns
        -------
        texture : ndarray
            RGBA texture array (height, width+1, 4) with uint8 values
        """
        # Create Earth-like color mapping
        # Normalize height map to [0, 1]
        h_min, h_max = self.height_map.min(), self.height_map.max()
        normalized = (
            (self.height_map - h_min) / (h_max - h_min)
            if h_max > h_min
            else np.zeros_like(self.height_map)
        )

        # Apply Earth-like colors manually
        # Define color stops (position, RGB)
        color_stops = [
            (0.0, np.array([26, 77, 122]) / 255),    # Deep ocean
            (0.3, np.array([45, 107, 163]) / 255),   # Ocean
            (0.4, np.array([74, 139, 194]) / 255),   # Shallow water
            (0.45, np.array([232, 214, 160]) / 255), # Beach
            (0.5, np.array([107, 142, 35]) / 255),   # Lowland
            (0.6, np.array([143, 188, 143]) / 255),  # Plains
            (0.7, np.array([160, 130, 109]) / 255),  # Hills
            (0.8, np.array([139, 115, 85]) / 255),   # Mountains
            (0.9, np.array([105, 105, 105]) / 255),  # High mountains
            (1.0, np.array([255, 255, 255]) / 255),  # Snow peaks
        ]

        # Initialize RGB array
        height, width = normalized.shape
        rgb = np.zeros((height, width, 3))

        # Apply color mapping
        for i in range(len(color_stops) - 1):
            pos1, color1 = color_stops[i]
            pos2, color2 = color_stops[i + 1]
            
            # Find pixels in this range
            mask = (normalized >= pos1) & (normalized < pos2)
            
            # Interpolate color
            t = (normalized - pos1) / (pos2 - pos1) if pos2 > pos1 else 0
            for c in range(3):
                rgb[:, :, c] = np.where(mask, 
                                         color1[c] + t * (color2[c] - color1[c]), 
                                         rgb[:, :, c])
        
        # Handle the last segment
        pos_last, color_last = color_stops[-1]
        mask = normalized >= pos_last
        for c in range(3):
            rgb[:, :, c] = np.where(mask, color_last[c], rgb[:, :, c])

        # Create RGBA array (flip vertically for correct orientation)
        rgba = np.dstack([rgb, np.ones((height, width))])
        rgba = np.flipud(rgba)

        # Add wrap-around column for seamless longitude transition
        rgba = np.concatenate([rgba, rgba[:, 0:1, :]], axis=1)

        # Convert to uint8
        return (rgba * 255).astype(np.uint8)


def generate_planet(width=800, height=400, num_faults=100):
    """
    Generate a fractal planet and return the texture array.
    
    Parameters
    ----------
    width : int
        Width of the texture map
    height : int
        Height of the texture map
    num_faults : int
        Number of fault lines to apply
        
    Returns
    -------
    texture : ndarray
        RGBA texture array (height, width+1, 4) with uint8 values
    """
    planet = FractalPlanet(width=width, height=height)
    planet.generate_terrain(num_faults=num_faults, initial_displacement=1.0, decay=1.0)
    return planet.get_texture_array()


def generate_multiple_planets(width=800, height=400, fault_counts=[100, 200, 300, 400, 500]):
    """
    Generate multiple fractal planets with different fault counts efficiently.
    Instead of regenerating from scratch, accumulates faults incrementally.
    
    Parameters
    ----------
    width : int
        Width of the texture map
    height : int
        Height of the texture map
    fault_counts : list of int
        List of fault counts to generate (must be in ascending order)
        
    Returns
    -------
    textures : list of ndarray
        List of RGBA texture arrays
    """
    textures = []
    planet = FractalPlanet(width=width, height=height)
    
    prev_count = 0
    for num_faults in fault_counts:
        # Add only the additional faults needed
        additional_faults = num_faults - prev_count
        if additional_faults > 0:
            planet.generate_terrain(num_faults=additional_faults, initial_displacement=1.0, decay=1.0)
        
        # Generate texture for current state
        texture = planet.get_texture_array()
        textures.append(texture)
        
        prev_count = num_faults
    
    return textures


if __name__ == "__main__":
    # For testing purposes
    texture = generate_planet(width=800, height=400, num_faults=100)
    print(f"Generated texture with shape: {texture.shape}")
    print(f"Texture dtype: {texture.dtype}")
    print(f"Texture value range: [{texture.min()}, {texture.max()}]")
