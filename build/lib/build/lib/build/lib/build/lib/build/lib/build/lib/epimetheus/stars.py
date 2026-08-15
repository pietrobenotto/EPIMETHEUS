import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
from astropy.stats import sigma_clip
from astropy.coordinates import SkyCoord
from astropy import units as u

from .io import Image 
from . import sewpy, config
from .check_dependencies import check_sextractor

class StarSelector:
    """
    A class to run SExtractor on astronomical images and interactively select stars
    based on flux concentration (size ratio) and magnitude criteria.
    """
    
    # Default SExtractor configuration
    DEFAULT_CONFIG = {
        "DETECT_MINAREA": 8,
        "DETECT_THRESH": 10,
        "ANALYSIS_THRESH": 10,
        "SEEING_FWHM": 0.08,
        "DEBLEND_NTHRESH": 32,
        "DEBLEND_MINCONT": 0.005,
        "FILTER": "N",
        "CLEAN": "Y",
        "CLEAN_PARAM": 1,
        "GAIN": 1,
        "BACK_SIZE": 64,
        "BACK_FILTERSIZE": 3,
        "BACKPHOTO_TYPE": "LOCAL",
        "MEMORY_OBJSTACK": 3000,
        "MEMORY_PIXSTACK": 3000000,
        "MEMORY_BUFSIZE": 8192
    }
    
    # Default SExtractor output parameters
    DEFAULT_PARAMS = [
        "X_IMAGE", "Y_IMAGE", 
        "ALPHA_J2000", "DELTA_J2000",
        "FLUX_APER(2)", "MAG_APER(2)", 
        "FLUX_MAX", "FLAGS"
    ]
    
    # Default apertures in arcseconds
    DEFAULT_APERTURES = np.array([0.16, 0.32])
    
    # Mapping from Image variance_type to SExtractor WEIGHT_TYPE
    VARIANCE_TYPE_MAP = {
        "var": "MAP_VAR",
        "rms": "MAP_RMS",
        "weight": "MAP_WEIGHT"
    }
    
    def __init__(self, 
                 image: Image,
                 apertures: np.ndarray = None,
                 sexpath: str = None,
                 use_variance: bool = True,
                 config_overrides: dict = None,
                 params_overrides: list = None,
                 cache_dir:str = None,
                 star_cat_filename:str = None ):
        """
        Initialise the StarSelector.
        
        Parameters
        ----------
        image : Image
            An Image object containing the FITS data, variance, and metadata.
        apertures : np.ndarray, optional
            Aperture diameters in arcseconds. Default is [0.16, 0.32].
        sexpath : str, optional
            Path to the SExtractor executable. Default is found trying, in this order, 'sex', 'sextractor', and 'source-extractor'.
        use_variance : bool, optional
            If True and variance is available in Image, use it as weight map.
            Default is True.
        config_overrides : dict, optional
            Dictionary of SExtractor configuration parameters to override defaults.
        params_overrides : list, optional
            List of additional SExtractor output parameters.
        cache_dir : str, optional
            Directory for caching detected objects files
        """
        # Store the Image object
        self.image = image
        
        # Extract relevant attributes from Image
        self.image_path = image.image_path
        self.pixel_scale = image.pixel_scale.to("arcsec").value
        self.mag_zeropoint = image.mag_zeropoint
        
        # Handle variance/weight image
        self.use_variance = use_variance and image.has_variance
        if self.use_variance:
            self.weight_image = image.variance_path
            self.weight_type = self.VARIANCE_TYPE_MAP.get(image.variance_type)
        else:
            self.weight_image = None
            self.weight_type = None
        
        # Other parameters
        self.apertures = apertures if apertures is not None else self.DEFAULT_APERTURES

        if sexpath is None:
            sexpath, __ = check_sextractor()
            if sexpath is None:
                raise ImportError("Sextractor not found in the system. If sextractor is installed, plese provide the sexpath attribute explicitely")

        self.sexpath = sexpath
        self.config_overrides = config_overrides or {}
        self.params_overrides = params_overrides or []
        
        # Derive catalogue name if not provided
        if cache_dir is None:
            cache_dir = config.CACHE_DIR

        self.cache_dir = cache_dir

        if star_cat_filename is None:
            star_cat_filename, __ = os.path.splitext(os.path.basename(image.image_path))

        self.cat_name = os.path.join(cache_dir, f"stars/{star_cat_filename}.cat")
        
        # Initialise attributes
        self.cat_objs = None
        self.cat_stars = None
        self.star_mask = None
        self.expected_ratio = None
        self.mag_limit = None
        self.mag_bright_limit = None
        self.obtained_ratio = None
        
        # Computed quantities
        self.flux_small = None
        self.flux_large = None
        self.mag_large = None
        self.size_ratio = None
    
    def _build_config(self) -> dict:
        """Build the SExtractor configuration dictionary."""
        # Start with defaults
        config = self.DEFAULT_CONFIG.copy()
        
        # Add apertures and pixel scale
        apertures_in_pixels = ",".join((self.apertures / self.pixel_scale).astype(str))
        config["PHOT_APERTURES"] = apertures_in_pixels
        config["PIXEL_SCALE"] = self.pixel_scale

        # Add magnitude zeropoint
        config["MAG_ZEROPOINT"] = self.mag_zeropoint
        
        # Apply user overrides
        config.update(self.config_overrides)
        
        return config
    
    def _build_params(self) -> list:
        """Build the SExtractor output parameters list."""
        params = self.DEFAULT_PARAMS.copy()
        params.extend(self.params_overrides)
        return params
    
    def run_sextractor(self, force_rerun: bool = False) -> Table:
        """
        Run SExtractor on the image.
        
        Parameters
        ----------
        force_rerun : bool, optional
            If True, rerun SExtractor even if catalogue exists. Default is False.
        
        Returns
        -------
        astropy.table.Table
            The SExtractor output catalogue.
        """
        # Check if catalogue already exists
        if os.path.exists(self.cat_name) and not force_rerun:
            print(f"Loading existing catalogue: {self.cat_name}")
            self.cat_objs = Table.read(self.cat_name, format="ascii")
            self._compute_flux_quantities()
            return self.cat_objs
        
        # Build configuration
        config = self._build_config()
        params = self._build_params()
        
        # Add weight image configuration if available
        if self.use_variance:
            config["WEIGHT_TYPE"] = self.weight_type
            config["WEIGHT_IMAGE"] = self.weight_image
        
        # Create SEW object and run
        sew = sewpy.SEW(sexpath=self.sexpath, params=params, config=config)
        
        try:
            print(f"Running SExtractor on {self.image_path}...")
            if self.use_variance:
                print(f"  Using weight image: {self.weight_image} ({self.weight_type})")
            out = sew(self.image_path)
            self.cat_objs = out["table"]
            
        except Exception as e:
            if self.use_variance:
                print(f"SExtractor failed with weight image: {e}")
                print("Retrying without weight image...")
                
                # Remove weight configuration and retry
                config.pop("WEIGHT_TYPE", None)
                config.pop("WEIGHT_IMAGE", None)
                
                sew = sewpy.SEW(sexpath=self.sexpath, params=params, config=config)
                out = sew(self.image_path)
                self.cat_objs = out["table"]
            else:
                raise e
        
        # Save catalogue
        directory = os.path.dirname(self.cat_name)
        os.makedirs(directory, exist_ok=True)

        self.cat_objs.write(self.cat_name, format="ascii", overwrite=True)
        print(f"Catalogue saved to: {self.cat_name}")
        
        # Compute flux quantities
        self._compute_flux_quantities()
        
        return self.cat_objs
    
    def _compute_flux_quantities(self):
        """Compute flux ratios and magnitudes from the catalogue."""
        self.flux_small = self.cat_objs['FLUX_APER']
        self.flux_large = self.cat_objs['FLUX_APER_1']
        self.mag_large = self.cat_objs['MAG_APER_1']
        self.size_ratio = self.flux_small / self.flux_large
    
    def plot_diagnostic(self, 
                        mag_lim: float = None,
                        show: bool = True,
                        save: bool = False,
                        output_path: str = None) -> None:
        """
        Create a diagnostic plot showing magnitude vs flux ratio.
        
        Parameters
        ----------
        mag_range : tuple, optional
            Magnitude limit for the x-axis.
        show : bool, optional
            If True, display the plot interactively. Default is True.
        save : bool, optional
            If True, save the plot to file. Default is False.
        output_path : str, optional
            Output path for the saved plot. If None, derived from cat_name.
        """
        if self.cat_objs is None:
            raise ValueError("No catalogue loaded. Run run_sextractor() first.")
        
        fig, ax = plt.subplots(figsize=(5, 5))
        
        # Plot all objects
        ax.scatter(self.mag_large, self.size_ratio, s=3, c='k', alpha=0.1, label='All objects')
        
        # Plot selected stars if available
        if self.star_mask is not None:
            ax.scatter(self.mag_large[self.star_mask], self.size_ratio[self.star_mask], 
                       s=3, c='dodgerblue', label='Selected stars', alpha=0.5)
            
            if self.obtained_ratio is not None:
                ax.axhline(y=self.obtained_ratio, color='b', linestyle='--', 
                           alpha=0.3, lw=0.5, label=f'Fitted ratio: {self.obtained_ratio:.3f}')
        
        # Plot expected ratio if set
        if self.expected_ratio is not None:
            ax.axhline(y=self.expected_ratio, color='r', linestyle=':', 
                       alpha=0.5, lw=1, label=f'Expected ratio: {self.expected_ratio:.3f}')
            ax.axhspan(self.expected_ratio - self.ratio_tolerance, self.expected_ratio + self.ratio_tolerance, 
                       alpha=0.1, color='red', label='Selection range')
        
        # Plot magnitude limit if set
        if self.mag_limit is not None:
            ax.axvline(x=self.mag_limit, color='green', linestyle='--', 
                       alpha=0.5, lw=1, label=f'Mag limit: {self.mag_limit:.1f}')
            
        if self.mag_bright_limit is not None and self.mag_bright_limit>-np.inf:
            ax.axvline(x=self.mag_bright_limit, color='green', linestyle='--', 
                       alpha=0.5, lw=1, label=f'Mag limit: {self.mag_bright_limit:.1f}')
        
        ax.set_xlabel(f'Magnitude ({self.apertures[1]:.2f}" aper)')
        ax.set_ylabel(f'Flux Ratio ({self.apertures[0]:.2f}" / {self.apertures[1]:.2f}")')
        if mag_lim is not None:
            ax.set_xlim(mag_lim, np.nanmin(self.mag_large) - 1)
        else:
            ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_ylim(0, 1)
        ax.legend(loc='lower right', fontsize=9, frameon=True)
        #ax.set_title(os.path.basename(self.image_path))
        
        plt.tight_layout()
        
        if save:
            if output_path is None:
                output_path = self.cat_name.replace(".cat", "_diagnostic.pdf")
            plt.savefig(output_path, dpi=150)
            print(f"Diagnostic plot saved to: {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def select_stars(self, 
                     expected_ratio: float, 
                     mag_limit: float,
                     mag_bright_limit = -np.inf,
                     ratio_tolerance: float = 0.05,
                     sigma_clip_sigma: float = 3.0,
                     sigma_clip_maxiters: int = 10) -> Table:
        """
        Select stars based on flux ratio and magnitude criteria.
        
        Parameters
        ----------
        expected_ratio : float
            Expected flux ratio (small/large aperture) for point sources.
        mag_limit : float
            Faint magnitude limit for star selection.
        ratio_tolerance : float, optional
            Tolerance around expected_ratio for initial selection. Default is 0.05.
        sigma_clip_sigma : float, optional
            Sigma threshold for outlier rejection. Default is 3.0.
        sigma_clip_maxiters : int, optional
            Maximum iterations for sigma clipping. Default is 10.
        
        Returns
        -------
        astropy.table.Table
            Catalogue of selected stars.
        """
        if self.cat_objs is None:
            raise ValueError("No catalogue loaded. Run run_sextractor() first.")
        
        self.expected_ratio = expected_ratio
        self.ratio_tolerance = ratio_tolerance
        self.mag_limit = mag_limit
        self.mag_bright_limit = mag_bright_limit
        
        # Initial selection based on magnitude and flux ratio
        self.star_mask = (
            (self.mag_large < mag_limit) & 
            (self.mag_large > mag_bright_limit) & 
            (self.size_ratio > expected_ratio - ratio_tolerance) & 
            (self.size_ratio < expected_ratio + ratio_tolerance)
        )
        
        self.obtained_ratio = expected_ratio
        
        # Sigma clipping to refine selection
        if np.sum(self.star_mask) > 0:
            clipped_data = sigma_clip(
                self.size_ratio[self.star_mask], 
                sigma=sigma_clip_sigma, 
                maxiters=sigma_clip_maxiters, 
                cenfunc='mean'
            )
            
            # Update obtained ratio and mask
            self.obtained_ratio = np.ma.mean(clipped_data)
            self.star_mask[self.star_mask] = ~clipped_data.mask
            
            print(f"Star selection: expected ratio = {expected_ratio:.4f}, "
                  f"obtained ratio = {self.obtained_ratio:.4f}")
            print(f"Selected {np.sum(self.star_mask)} stars")
        else:
            print("Warning: No valid stars found with the specified criteria.")
        
        # Create star catalogue
        self.cat_stars = self.cat_objs[self.star_mask]
        
        return self.cat_stars
    
    def save_star_catalogue(self, output_path: str = None) -> None:
        """
        Save the selected star catalogue to file.
        
        Parameters
        ----------
        output_path : str, optional
            Output path for the catalogue. If None, derived from cat_name.
        """
        if self.cat_stars is None:
            raise ValueError("No stars selected. Run select_stars() first.")
        
        if output_path is None:
            output_path = self.cat_name.replace(".cat", "_stars.cat")
        
        self.cat_stars.write(output_path, format="ascii", overwrite=True)
        print(f"Star catalogue saved to: {output_path}")
    
    def save_diagnostic_plot(self, output_path: str = None) -> None:
        """
        Save the diagnostic plot to file.
        
        Parameters
        ----------
        output_path : str, optional
            Output path for the plot. If None, derived from cat_name.
        """
        self.plot_diagnostic(show=False, save=True, output_path=output_path)

    # =========================================================================
    # Multi-Catalogue Merging Methods
    # =========================================================================
    

class MergedCatalogue:
    
    def __init__(self,
                catalogue_files: list | dict,
                field_names: list = [],
                duplicate_radius: float = None,
                sort_by_mag: bool = True,
                mag_column: str = 'MAG_APER_1') -> Table:
        """
        Merge multiple star catalogues, removing duplicates.
        
        Parameters
        ----------
        catalogue_files : list
            List of paths to star catalogue files.
        field_names : list,
            List of field names for each catalogue (same order of catalagues).
        duplicate_radius : float, optional
            Radius in arcseconds for duplicate matching. Default is None.
            If None, all the stars are kept
        sort_by_mag : bool, optional
            If True, sort final catalogue by magnitude. Default is True.
        mag_column : str, optional
            Column name for magnitude sorting. Default is 'MAG_APER_1'.
        """

        if isinstance(catalogue_files, list):
            if len(catalogue_files) !=len(field_names):
                raise ValueError("The length of field_names must be equal to the catalogue_files length")
            self.catalogue_files = catalogue_files
            self.field_names = field_names
        else:
            #is a dict
            self.field_names = list(catalogue_files.keys())
            self.catalogue_files = list(catalogue_files.values())

        self.duplicate_radius = duplicate_radius
        self.sort_by_mag = sort_by_mag
        self.mag_column = mag_column

        self._merge_catalogues()

        self.n_rows = len(self.master_catalog)



    def _merge_catalogues(self):
        """
        Returns
        -------
        astropy.table.Table
            Merged catalogue with duplicates removed.
        """
        
        temp_catalogs = []
        
        print("Building Master Catalog...")
        
        for i, cat_stars in enumerate(self.catalogue_files):
        
            # Determine origin label
            if self.field_names is not None and i < len(self.field_names):
                field = self.field_names[i]
            else:
                raise ValueError("Please call merge_catalogues with proper field_names")

            # Add metadata columns
            cat_stars['origin_field'] = field
            
            temp_catalogs.append(cat_stars)
            print(f"  Loaded {len(cat_stars)} stars from {i+1}th catalogue")
        
        if len(temp_catalogs) == 0:
            raise ValueError("No catalogues found!")
        
        # Stack into one big table
        self.master_catalog = vstack(temp_catalogs)
        print(f"Total stars before duplicate removal: {len(self.master_catalog)}")
        
        # Remove duplicates
        if self.duplicate_radius is not None:
            self._remove_duplicates(self.duplicate_radius)
        
        # Sort by magnitude if requested
        if self.sort_by_mag and self.mag_column in self.master_catalog.colnames:
            self.master_catalog.sort(self.mag_column)
        
        # Compute aperture ratio statistics
        self._compute_master_statistics()
        
        if self.duplicate_radius is not None:
            print(f"Unique stars after duplicate removal: {len(self.master_catalog)}")
        
        return self.master_catalog
    
    def _remove_duplicates(self, radius: float = 1.0) -> None:
        """
        Remove duplicate stars within a given radius.
        
        Parameters
        ----------
        radius : float, optional
            Matching radius in arcseconds. Default is 1.0.
        """
        if self.master_catalog is None or len(self.master_catalog) <= 1:
            return
        
        coords = SkyCoord(
            self.master_catalog["ALPHA_J2000"], 
            self.master_catalog["DELTA_J2000"], 
            unit="deg"
        )
        
        keep_mask = np.ones(len(self.master_catalog), dtype=bool)
        
        # Find all pairs within radius
        idx1, idx2, d2d, _ = coords.search_around_sky(coords, radius * u.arcsec)
        
        # Remove duplicates (keep the first occurrence, i.e., where idx1 < idx2)
        duplicates = idx1[idx1 > idx2]
        keep_mask[duplicates] = False
        
        n_removed = np.sum(~keep_mask)
        if n_removed > 0:
            print(f"  Removed {n_removed} duplicate stars (within {radius} arcsec)")
        
        self.master_catalog = self.master_catalog[keep_mask]
    
    def _compute_master_statistics(self) -> None:
        """Compute aperture ratio statistics for the master catalogue."""
        if self.master_catalog is None:
            return
        
        if 'FLUX_APER' in self.master_catalog.colnames and 'FLUX_APER_1' in self.master_catalog.colnames:
            ratio = self.master_catalog["FLUX_APER"] / self.master_catalog["FLUX_APER_1"]
            
            # Filter out invalid values
            valid = np.isfinite(ratio) & (ratio > 0) & (ratio < 1)
            
            if np.sum(valid) > 0:
                self.median_ratio_apertures = np.median(ratio[valid])
                self.std_ratio_apertures = np.std(ratio[valid])
                print(f"  Aperture flux ratio: median = {self.median_ratio_apertures:.4f}, "
                      f"std = {self.std_ratio_apertures:.4f}")
    
    
    def save_master_catalogue(self, output_path: str = None) -> None:
        """
        Save the master catalogue to file.
        
        Parameters
        ----------
        output_path : str, optional
            Output path. If None, derived from filter_name.
        """
        if self.master_catalog is None:
            raise ValueError("No master catalogue. Run merge_catalogues() first.")
        
        if output_path is None:
                output_path = os.path.join(config.CACHE_DIR, "master_stars.cat")
        
        self.master_catalog.write(output_path, format="ascii", overwrite=True)
        print(f"Master catalogue saved to: {output_path}")