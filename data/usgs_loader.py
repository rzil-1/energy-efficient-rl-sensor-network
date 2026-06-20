import os
import numpy as np
from scipy.ndimage import zoom, gaussian_filter

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("Warning: rasterio not installed. GeoTIFF loading will be disabled.")

class TerrainLoader:
    """
    A pipeline to ingest real-world USGS elevation data (GeoTIFF)
    or generate realistic simulated terrain if data is missing.
    """
    def __init__(self, target_grid_size=(50, 50)):
        """
        Args:
            target_grid_size (tuple): The (height, width) of the grid expected by the RL environment.
        """
        self.target_grid_size = target_grid_size

    def load_usgs_geotiff(self, file_path):
        """
        Loads a USGS GeoTIFF, normalizes it, and rescales it to the target grid size.
        """
        if not RASTERIO_AVAILABLE:
            raise ImportError("rasterio is required to load GeoTIFFs. Please install it.")
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"GeoTIFF file not found at {file_path}")

        print(f"Loading USGS terrain data from {file_path}...")
        with rasterio.open(file_path) as src:
            # Read the first band (elevation)
            elevation_data = src.read(1).astype(np.float32)
            
            # Handle nodata values
            nodata = src.nodata
            if nodata is not None:
                elevation_data[elevation_data == nodata] = np.nan
                # Fill NaNs with the lowest valid elevation (simplification)
                min_valid = np.nanmin(elevation_data)
                elevation_data[np.isnan(elevation_data)] = min_valid
                
        # Normalize to [0, 1]
        min_el = np.min(elevation_data)
        max_el = np.max(elevation_data)
        if max_el > min_el:
            normalized_data = (elevation_data - min_el) / (max_el - min_el)
        else:
            normalized_data = elevation_data

        # Rescale to target grid size
        current_shape = normalized_data.shape
        zoom_factors = (
            self.target_grid_size[0] / current_shape[0],
            self.target_grid_size[1] / current_shape[1]
        )
        rescaled_data = zoom(normalized_data, zoom_factors, order=1) # Bilinear interpolation
        
        return rescaled_data

    def generate_simulated_terrain(self, seed=None):
        """
        Generates a realistic terrain matrix using Gaussian smoothed noise
        if no USGS data is available. This prevents the training pipeline from breaking.
        """
        if seed is not None:
            np.random.seed(seed)
            
        print("Generating simulated realistic terrain...")
        # Create random noise
        base_noise = np.random.rand(*self.target_grid_size)
        
        # Apply gaussian filter to smooth the noise into "hills" and "valleys"
        # Sigma controls the size of the hills
        smoothed_terrain = gaussian_filter(base_noise, sigma=3.0)
        
        # Normalize to [0, 1]
        min_el = np.min(smoothed_terrain)
        max_el = np.max(smoothed_terrain)
        normalized_terrain = (smoothed_terrain - min_el) / (max_el - min_el)
        
        return normalized_terrain

    def get_terrain(self, file_path=None, seed=None):
        """
        Main interface. Tries to load GeoTIFF if path provided, else generates simulated.
        """
        if file_path and os.path.exists(file_path):
            try:
                return self.load_usgs_geotiff(file_path)
            except Exception as e:
                print(f"Failed to load GeoTIFF: {e}. Falling back to simulation.")
                return self.generate_simulated_terrain(seed=seed)
        else:
            if file_path:
                print(f"File {file_path} not found. Falling back to simulation.")
            return self.generate_simulated_terrain(seed=seed)

if __name__ == "__main__":
    # Test the loader
    loader = TerrainLoader(target_grid_size=(20, 20))
    terrain = loader.get_terrain(seed=42)
    print("Terrain shape:", terrain.shape)
    print("Terrain min/max:", np.min(terrain), np.max(terrain))
    print("Top-left 3x3 patch:\\n", terrain[:3, :3])
