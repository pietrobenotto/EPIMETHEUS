import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from photutils.centroids import centroid_2dg
from photutils.aperture import CircularAperture
from scipy.ndimage import affine_transform
from skimage.measure import block_reduce
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
from matplotlib.patches import Circle
from tqdm import tqdm
import os
import warnings


from .stars import MergedCatalogue
from .io import ImagesCollection
from .config import CACHE_DIR

def harmonic_mean_weight(x, axis):
    #mask negative and null values
    x_safe = np.where(x > 0, x, np.nan)

    with np.errstate(divide='ignore', invalid='ignore'):
        inv_sum = np.nansum(1.0 / x_safe, axis=axis)
        result = 1.0 / inv_sum
        # weight = 0
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def get_combined_transform(shape_in, shape_out, shift, angle_deg, scale_factor):
    """
    Affine matrix combining: Upscaling -> Shift -> Rotation.
    Maps OUTPUT (high res, rotated) coordinates -> INPUT (low res) coordinates.
    
    Forward operations (on image):
        1. Upscale by scale_factor (center preserved)
        2. Shift by 'shift' in upscaled, UN-ROTATED coordinates
        3. Rotate by angle_deg around output center
    
    Parameters:
    -----------
    shape_in : tuple
        Input image shape (h, w)
    shape_out : tuple
        Output image shape (h, w)
    shift : tuple
        (shift_y, shift_x) in HIGH-RES (upscaled) pixels, BEFORE rotation
    angle_deg : float
        Rotation angle in degrees
    scale_factor : float
        Upscaling factor (output_size / input_size)
    
    Returns:
    --------
    matrix : 2x2 array
        Affine transformation matrix
    offset : 1D array
        Affine offset vector
    """
    # Centers
    center_in = (np.array(shape_in) - 1) / 2.0
    center_out = (np.array(shape_out) - 1) / 2.0
    
    # Inverse rotation matrix (R^T)
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    rot_inv = np.array([[c, s], [-s, c]])  # R^T = inverse rotation
    
    # Inverse scaling (output -> input)
    scale_inv = np.array([[1/scale_factor, 0], [0, 1/scale_factor]])
    
    # Combined matrix: first un-rotate, then un-scale
    # M = (1/scale_factor) * R^T
    matrix = scale_inv @ rot_inv  # or equivalently: rot_inv / scale_factor
    
    # Shift in low-res coordinates (just scale, NO rotation!)
    shift_low = np.array(shift) / scale_factor
    
    # Offset: we want center_out (after shift=0) to map to center_in
    # Full inverse: x_in = matrix @ x_out + center_in - matrix @ center_out - shift_low
    offset = center_in - matrix @ center_out - shift_low  # <-- NO rot_inv @ here!
    
    return matrix, offset


def process_cutout(data, shift, angle = 0.0, scale_factor=1, output_shape=None, order=3):
    """
    Apply upscaling + shift + rotation in ONE pass.
    
    Operations (in forward order):
        1. Upscale by scale_factor
        2. Shift by 'shift' (in upscaled pixels, NOT rotated)
        3. Rotate by 'angle' degrees around center
    
    Parameters:
    -----------
    data : 2D array
        Input image
    shift : tuple
        (shift_y, shift_x) in upscaled (high-res) pixels, before rotation
    angle : float
        Rotation angle in degrees, default is 0
    scale_factor : float
        Upscaling factor, default is 1
    output_shape : tuple
        Output image shape, if None the original shape will be used
    order : int
        Spline interpolation order
    
    Returns:
    --------
    result : 2D array
        Transformed image
    """
    if output_shape is None:
        output_shape = data.shape

    matrix, offset = get_combined_transform(
        shape_in=data.shape,
        shape_out=output_shape,
        shift=shift,
        angle_deg=angle,
        scale_factor=scale_factor
    )


    #import sextractor.lanczos as lanczos
    #result = lanczos.affine_transform_lanczos(data, matrix, offset, output_shape, a=order)

    result = affine_transform(
        data, 
        matrix, 
        offset=offset,
        output_shape=output_shape,
        order=order,
        mode='constant',
        cval=0.0
    )
    
    """plt.subplots(1,2,sharex=True,sharey=True)
    plt.subplot(121)
    plt.imshow(result,origin="lower")
    plt.subplot(122)
    plt.imshow(result_old,origin="lower")
    plt.show()"""
    
    
    return result

def align_and_rotate_star(
    cutout: np.ndarray,
    cutout_wht: Optional[np.ndarray],
    shift: Tuple[float, float],
    best_pa: float,
    scale_factor: int,
    upsample_factor: int,
    final_size: np.ndarray,
    final_angle: float = 0.0,
    rotation: Optional[float] = None,
    fine_rotation: bool = False,
    order: int = 3
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Align and rotate a star cutout.
    
    Applies shift, rotation, and resampling in a single transformation.
    
    Parameters
    ----------
    cutout : np.ndarray
        Original star cutout.
    cutout_wht : np.ndarray or None
        Weight map (inverse variance).
    shift : Tuple[float, float]
        (shift_y, shift_x) in native pixels.
    best_pa : float
        Best position angle of the image in degrees.
    scale_factor : int
        Scale factor between image resolution and final resolution.
    upsample_factor : int
        Upsampling factor for high-resolution processing.
    final_size : np.ndarray
        Size of the final cutout (height, width).
    final_angle : float, optional
        Target orientation angle in degrees. Default is 0.0.
    rotation : float or None, optional
        Measured rotation angle in degrees.
    fine_rotation : bool, optional
        If True, include the measured rotation correction. Default is False.
    order : int, optional
        Interpolation order (1-5). Default is 3.
    
    Returns
    -------
    Tuple[np.ndarray, Optional[np.ndarray]]
        (aligned_cutout, aligned_weight) at final resolution.
    """
    high_res_factor = max(upsample_factor, scale_factor)
    upsample_ratio = high_res_factor / scale_factor
    output_shape = tuple(final_size * high_res_factor)
    
    rot_angle = -best_pa + final_angle
    if fine_rotation and rotation is not None:
        rot_angle += rotation

    # Transform science image
    rotated_cutout = process_cutout(
        data=cutout,
        shift=shift,
        angle=rot_angle,
        scale_factor=upsample_ratio,
        output_shape=output_shape,
        order=order
    )
    reduced_image = block_reduce(rotated_cutout, high_res_factor)
    
    # Transform weight map
    reduced_wht = None
    if cutout_wht is not None:
        rotated_wht = process_cutout(
            data=cutout_wht,
            shift=shift,
            angle=rot_angle,
            scale_factor=upsample_ratio,
            output_shape=output_shape,
            order=1
        )
        reduced_wht = block_reduce(rotated_wht, high_res_factor, func=harmonic_mean_weight)
    
    return reduced_image, reduced_wht


def normalize_cutout(
    cutout: np.ndarray,
    cutout_wht: Optional[np.ndarray],
    norm_radius: float,
    final_resolution: float
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Normalize a cutout by its flux within a circular aperture.
    
    Parameters
    ----------
    cutout : np.ndarray
        2D cutout image.
    cutout_wht : np.ndarray or None
        Weight map (inverse variance).
    norm_radius : float
        Aperture radius in arcseconds.
    final_resolution : float
        Pixel scale in arcseconds/pixel.
    
    Returns
    -------
    Tuple[np.ndarray, Optional[np.ndarray]]
        (normalized_cutout, normalized_weight).
    """
    from photutils.aperture import CircularAperture
    
    center = (np.array(cutout.shape) - 1) / 2.0
    r_px = norm_radius / final_resolution
    aperture = CircularAperture(center, r=r_px)
    flux = aperture.do_photometry(cutout)[0][0]
    
    if flux <= 0:
        flux = 1.0  # Fallback to avoid division by zero
    
    norm_cutout = cutout / flux
    
    norm_wht = None
    if cutout_wht is not None:
        norm_wht = cutout_wht * flux**2
    
    return norm_cutout, norm_wht

@dataclass
class StarContainer:
    """
    Container for a single star's cutout data and metadata.
    
    Parameters
    ----------
    image_name : str
        Name/identifier of the source image.
    x : float
        Centroid position in the image, x coordinate.
    y : float
        Centroid position in the image, y coordinate.
    alpha : float, optional
        Right ascension in degrees.
    delta : float, optional
        Declination in degrees.
    mag : float, optional
        Magnitude of the star.
    cutout : np.ndarray, optional
        2D array containing the star cutout.
    scale_factor : int, optional
        Scale factor between image resolution and final resolution.
    best_pa : float, optional
        Best position angle in degrees.
    cutout_wht : np.ndarray, optional
        2D array containing the weight/inverse variance cutout.
    cutout_elaborated : np.ndarray, optional
        2D array containing the aligned and rotated cutout.
    cutout_wht_elaborated : np.ndarray, optional
        2D array containing the aligned and rotated weight map.
    rotation : float, optional
        Measured rotation angle in degrees.
    shift : Tuple[float, float], optional
        Measured (shift_y, shift_x) in pixels.
    background : float, optional
        Subtracted background value.
    is_valid : int
        Quality flag: 1 = passed, 0 = failed, -1 = unchecked.
    mask : bool
        If True, star is excluded from stacking. Default is False.
    failure_reason : str, optional
        Description of why the star was marked invalid/masked.
    """
    alpha: float
    delta: float
    image_name: str
    x: Optional[float] = None
    y: Optional[float] = None
    mag: Optional[float] = None
    cutout: Optional[np.ndarray] = None
    scale_factor: Optional[int] = None
    best_pa: Optional[float] = None
    cutout_wht: Optional[np.ndarray] = None
    cutout_elaborated: Optional[np.ndarray] = None
    cutout_wht_elaborated: Optional[np.ndarray] = None
    rotation: Optional[float] = None
    shift: Optional[Tuple[float, float]] = None
    background: Optional[float] = None
    is_valid: int = -1
    mask: bool = False
    failure_reason: Optional[str] = None

    def __repr__(self) -> str:
        """Return a concise string representation."""
        cutout_shape = self.cutout.shape if self.cutout is not None else None
        has_wht = self.cutout_wht is not None
        
        if self.is_valid == 1 and not self.mask:
            status = "Good"
        elif self.is_valid == 0 or self.mask:
            status = f"Bad: ({self.failure_reason})"
        else:
            status = "Unknown"
        
        shift_str = f"({self.shift[0]:.2f}, {self.shift[1]:.2f})" if self.shift else "None"
        rot_str = f"{self.rotation:.2f}°" if self.rotation is not None else "None"
        
        return (
            f"StarContainer(\n"
            f"  image='{self.image_name}', pos=({self.x:.1f}, {self.y:.1f}),\n"
            f"  RA={self.alpha:.5f}°, Dec={self.delta:.5f}°, mag={self.mag:.2f},\n"
            f"  cutout={cutout_shape}, has_wht={has_wht}, scale={self.scale_factor},\n"
            f"  PA={self.best_pa:.1f}°, rotation={rot_str}, shift={shift_str},\n"
            f"  status={status}\n"
            f")"
        )


class StarCutouts:
    """
    Handles extraction and processing of star cutouts from astronomical images.
    
    This class manages the extraction of stellar cutouts from a collection of images,
    including background subtraction, centroid measurement, and rotation angle
    determination for PSF characterisation.
    
    Parameters
    ----------
    catalogue : MergedCatalogue
        Merged catalogue containing star positions and metadata.
    image_collection : ImagesCollection
        Collection of Image objects from which to extract cutouts.
    name : str
        Name identifier for this ePSF extraction (used for cache files).
    final_resolution : float
        Target pixel scale in arcseconds/pixel for the final ePSF.
    final_size : tuple or np.ndarray, optional
        Size of the final cutout in pixels (height, width). Default is (205, 205).
    upsample_factor : float, optional
        Upsampling factor for high-resolution processing. Default is 1.
    cache : str, optional
        Path to cache directory. Default is "./epi".
    
    Attributes
    ----------
    name : str
        Name identifier for this extraction.
    cache_dir : str
        Path to cache directory.
    catalogue_path : str
        Path to the cached catalogue file.
    master_catalogue : astropy.table.Table
        The merged star catalogue.
    stars : list
        List of StarContainer objects.
    
    Raises
    ------
    ValueError
        If final_resolution is not an integer multiple of each image's pixel scale.
    """
    
    # Default parameters
    DEFAULT_CENTROID_RADIUS = 7
    DEFAULT_MAX_SHIFT = 3.0
    DEFAULT_SIGMA_CLIP = 2.0
    DEFAULT_SIGMA_MAXITERS = 10
    
    def __init__(self,
                 catalogue: MergedCatalogue,
                 image_collection: ImagesCollection,
                 name: str,
                 final_resolution: float,
                 cutout_shape: Tuple[int, int] | np.ndarray = (205, 205),
                 upsample_factor: float = 1,
                 cache: str = CACHE_DIR):
        
        # Store name and cache configuration
        self.name = name
        self.cache_dir = cache
        self.catalogue_dir = os.path.join(cache, "epsf_cat")
        self.catalogue_path = os.path.join(self.catalogue_dir, f"{name}.cat")
        
        # Create cache directory if needed
        os.makedirs(self.catalogue_dir, exist_ok=True)
        
        # Store catalogue data
        self.master_catalogue = catalogue.master_catalog
        self.median_ratio_apertures = catalogue.median_ratio_apertures
        self.std_ratio_apertures = catalogue.std_ratio_apertures
        
        # Store image collection and resolution parameters
        self.image_collection = image_collection
        self.final_resolution = final_resolution
        self.final_size = np.asarray(cutout_shape)
        self.upsample_factor = upsample_factor
        self.image_names = self.image_collection.image_names
        
        # Compute pixel scales and scale factors
        self.pixel_scales: Dict[str, float] = {}
        self.scale_factors: Dict[str, int] = {}
        
        for image_name in self.image_names:
            self.pixel_scales[image_name] = float(self.image_collection.data[image_name].pixel_scale.value)
            scale_factor = float(self.final_resolution / self.pixel_scales[image_name])
            
            if abs(scale_factor - round(scale_factor)) < 1e-6:
                self.scale_factors[image_name] = int(round(scale_factor))
                
            else:
                raise ValueError(
                    f"final_resolution must be an integer multiple of each image resolution. "
                    f"final_resolution: {final_resolution}, "
                    f"pixel_scale of {image_name} = {self.pixel_scales[image_name]}, "
                    f"ratio = {scale_factor}"
                )
        
        # Initialize stars list
        self.stars: list[Optional[StarContainer]] = [None] * len(self.master_catalogue)
        
        # Load cached catalogue if exists
        self._load_catalogue()
    
    def _load_catalogue(self) -> bool:
        """
        Load star metadata from cached catalogue file.
        
        Updates the stars list with rotation, is_valid, and mask values
        from the cached file if it exists.
        
        Returns
        -------
        bool
            True if catalogue was loaded, False otherwise.
        """
        if not os.path.exists(self.catalogue_path):
            print(f"No cached catalogue found at {self.catalogue_path}")
            return False
        
        try:
            cached = Table.read(self.catalogue_path, format="ascii.ecsv")
            print(f"Loading cached catalogue: {self.catalogue_path}")
            
            # Create index mapping by coordinates
            cached_coords = {
                (row['alpha'], row['delta']): row for row in cached
            }
            
            n_matched = 0
            for i, master_row in enumerate(self.master_catalogue):
                key = (master_row['ALPHA_J2000'], master_row['DELTA_J2000'])
                
                if key in cached_coords:
                    cached_row = cached_coords[key]
                    
                    # Update with cached values
                    star = self.stars[i]
                    if star is not None:
                        star.rotation = cached_row['rotation'] if cached_row['rotation'] != -999 else None
                        star.is_valid = int(cached_row['is_valid'])
                        star.mask = bool(cached_row['mask'])
                        star.failure_reason = cached_row['failure_reason'] if cached_row['failure_reason'] != '' else None
                        n_matched += 1
                    else:
                        self.stars[i] = StarContainer(
                            alpha = cached_row["alpha"],
                            delta = cached_row["delta"],
                            image_name = cached_row["image_name"],
                            mag = cached_row['mag'],
                            rotation = cached_row['rotation'] if cached_row['rotation'] != -999 else None,
                            is_valid = int(cached_row['is_valid']),
                            mask = bool(cached_row['mask']),
                            failure_reason = cached_row['failure_reason'] if cached_row['failure_reason'] != '' else None
                        )
                        n_matched += 1
            
            print(f"Loaded {n_matched}/{len(self.master_catalogue)} stars from cache")
            return True
            
        except Exception as e:
            print(f"Warning: Failed to load catalogue: {e}")
            return False
    
    def save_catalogue(self) -> str:
        """
        Save star metadata to catalogue file.
        
        Saves alpha, delta, magnitude, rotation, is_valid, mask, and 
        failure_reason for all extracted stars.
        
        Returns
        -------
        str
            Path to the saved catalogue file.
        """
        # Prepare data columns
        data = {
            'alpha': [],
            'delta': [],
            'mag': [],
            'image_name': [],
            'rotation': [],
            'is_valid': [],
            'mask': [],
            'failure_reason': []
        }
        
        for star in self.stars:
            if star is not None:
                data['alpha'].append(star.alpha)
                data['delta'].append(star.delta)
                data['mag'].append(star.mag)
                data['image_name'].append(star.image_name)
                data['rotation'].append(star.rotation if star.rotation is not None else -999)
                data['is_valid'].append(star.is_valid)
                data['mask'].append(int(star.mask))
                data['failure_reason'].append(star.failure_reason if star.failure_reason else '')
        
        # Create and save table
        cat = Table(data)
        cat.write(self.catalogue_path, format="ascii.ecsv", overwrite=True)
        
        print(f"Catalogue saved: {self.catalogue_path} ({len(cat)} stars)")
        return self.catalogue_path
        
    # =========================================================================
    # Cutout Extraction
    # =========================================================================
    
    def extract_cutout_ith_star(self, 
                                 row_index: int, 
                                 magnitude_field: str = "MAG_APER_1",
                                 force_extraction = False) -> StarContainer:
        """
        Extract the cutout for the i-th star in the catalogue.
        
        Parameters
        ----------
        row_index : int
            Index of the star in the master catalogue.
        magnitude_field : str, optional
            Column name for the magnitude. Default is "MAG_APER_1".
        force_extraction : bool, optional
            Extract the star again even it it's already extracted. Overwrite all values. 
        Returns
        -------
        StarContainer
            Container with the extracted cutout and metadata.
        
        Raises
        ------
        IndexError
            If row_index is out of bounds.
        """
        if self.stars[row_index] is not None and self.stars[row_index].cutout is not None and not force_extraction:
            #The star is already extracted
            return self.stars[row_index]
        
        image_name = self.master_catalogue['origin_field'][row_index]
        ra_star = self.master_catalogue['ALPHA_J2000'][row_index]
        dec_star = self.master_catalogue['DELTA_J2000'][row_index]
        x_star = self.master_catalogue['X_IMAGE'][row_index]
        y_star = self.master_catalogue['Y_IMAGE'][row_index]
        magnitude_star = self.master_catalogue[magnitude_field][row_index]
        
        image_obj = self.image_collection.data[image_name]
        fits_file_data = image_obj.image_data
        wcs_obj = image_obj.wcs
        
        star_coord = SkyCoord(ra_star, dec_star, unit="deg")
        cutout_shape = tuple(self.final_size * self.scale_factors[image_name])
        
        # Extract science cutout
        cutout = Cutout2D(
            fits_file_data, 
            star_coord, 
            size=cutout_shape, 
            wcs=wcs_obj, 
            copy=True
        ).data
        
        # Extract variance/weight cutout if available
        cutout_wht = None
        if image_obj.has_variance:
            var_data = image_obj.var_data
            cutout_var = Cutout2D(
                var_data, 
                star_coord, 
                size=cutout_shape, 
                wcs=wcs_obj, 
                copy=True
            ).data
            
            var_type = image_obj.variance_type

            cutout_wht = np.zeros_like(cutout_var)
            valid_var = np.isfinite(cutout_var) & (cutout_var>0)
            
            if var_type == "var":
                cutout_wht[valid_var] = 1.0 / cutout_var[valid_var]
            elif var_type == "rms":
                cutout_wht[valid_var] = 1.0 / np.power(cutout_var[valid_var], 2)
            elif var_type == "weight":
                cutout_wht[valid_var] = cutout_var[valid_var]
            else:
                raise ValueError(f"Invalid variance_type: {var_type}")


        if self.stars[row_index] is None or force_extraction:
            star = StarContainer(
                image_name=image_name,
                alpha=ra_star,
                delta=dec_star,
                x=x_star,
                y=y_star,
                mag=magnitude_star,
                cutout=cutout,
                cutout_wht=cutout_wht,
                scale_factor=self.scale_factors[image_name],
                best_pa=image_obj.best_PA
            )

            self.stars[row_index] = star

        else:
            star = self.stars[row_index]
            star.x = x_star
            star.y = y_star
            star.cutout = cutout
            star.cutout_wht = cutout_wht
            star.scale_factor = self.scale_factors[image_name]
            star.best_pa = image_obj.best_PA
        
        return star
    
    def extract_all_cutouts(self, 
                            magnitude_field: str = "MAG_APER_1",
                            force_extraction = False) -> None:
        """
        Extract cutouts for all stars in the catalogue.
        
        Parameters
        ----------
        magnitude_field : str, optional
            Column name for the magnitude. Default is "MAG_APER_1".
        verbose : bool, optional
            If True, print progress. Default is True.
        force_extraction : bool, optional
            Extract the stars again even if they are already extracted. Overwrite all values. 
        """
        n_stars = len(self.master_catalogue)
        
        for i in tqdm(range(n_stars)):
            try:
                self.extract_cutout_ith_star(i, magnitude_field, force_extraction=force_extraction)
            except Exception as e:
                print(f"Warning: Failed to extract cutout for star {i}: {e}")
                # Create an invalid StarContainer
                self.stars[i] = StarContainer(
                    image_name="",
                    alpha=self.master_catalogue['ALPHA_J2000'][i],
                    delta=self.master_catalogue['DELTA_J2000'][i],
                    mag=np.nan,
                    cutout=np.array([]),
                    scale_factor=1,
                    is_valid=False,
                    failure_reason=str(e)
                )
    
    # =========================================================================
    # Background Subtraction
    # =========================================================================
    
    def subtract_background(self, 
                            row_index: int,
                            sigma: float = None,
                            maxiters: int = None,
                            method: str = "median") -> float:
        """
        Subtract background from a star cutout using sigma-clipped statistics.
        
        Parameters
        ----------
        row_index : int
            Index of the star in the catalogue.
        sigma : float, optional
            Sigma threshold for clipping. Default is DEFAULT_SIGMA_CLIP (2.0).
        maxiters : int, optional
            Maximum iterations for sigma clipping. Default is DEFAULT_SIGMA_MAXITERS (10).
        method : str, optional
            Background estimation method: "median" or "mean". Default is "median".
        
        Returns
        -------
        float
            The subtracted background value.
        
        Raises
        ------
        ValueError
            If the star cutout has not been extracted yet.
        """
        if sigma is None:
            sigma = self.DEFAULT_SIGMA_CLIP
        if maxiters is None:
            maxiters = self.DEFAULT_SIGMA_MAXITERS
        
        star = self.stars[row_index]
        
        if star is None:
            raise ValueError(f"Star {row_index} has not been extracted. Run extract_cutout_ith_star first.")
        
        #background already computed
        if star.background is not None:
            return star.background
        
        # Compute sigma-clipped statistics
        mean_val, median_val, std_val = sigma_clipped_stats(
            star.cutout, 
            sigma=sigma, 
            maxiters=maxiters
        )
        
        # Select background value based on method
        if method == "median":
            background = median_val
        elif method == "mean":
            background = mean_val
        else:
            raise ValueError(f"Unknown method: {method}. Use 'median' or 'mean'.")
        
        # Subtract background
        star.cutout = star.cutout - background
        star.background = background

        if star.cutout_wht is None:
            star.cutout_wht = np.full_like(star.cutout,1/std_val**2)
        
        return background
    
    def subtract_background_all(self,
                                 sigma: float = None,
                                 maxiters: int = None,
                                 method: str = "median") -> None:
        """
        Subtract background from all star cutouts.
        
        Parameters
        ----------
        sigma : float, optional
            Sigma threshold for clipping.
        maxiters : int, optional
            Maximum iterations for sigma clipping.
        method : str, optional
            Background estimation method.
        verbose : bool, optional
            If True, print progress.
        """
        n_stars = len(self.stars)
        
        for i in tqdm(range(n_stars)):
            if self.stars[i] is not None:
                self.subtract_background(i, sigma, maxiters, method)
    
    # =========================================================================
    # Shift / Centroid Computation
    # =========================================================================
    
    
    def compute_shift_gaussian(self,
                               row_index: int,
                               centroid_radius: int = None,
                               max_shift: float = None,
                               force_recomputation: bool = False) -> Tuple[float, float]:
        """
        Compute the centroid shift using a 2D Gaussian fit.
        
        Fits a 2D Gaussian to the central region of the star cutout to determine
        the sub-pixel shift required to centre the star.
        
        Parameters
        ----------
        row_index : int
            Index of the star in the catalogue.
        centroid_radius : int, optional
            Radius in pixels (at image scale) for centroid fitting.
            Default is DEFAULT_CENTROID_RADIUS * scale_factor.
        max_shift : float, optional
            Maximum allowed shift in pixels (at image scale).
            Default is DEFAULT_MAX_SHIFT * scale_factor.
        force_recomputation : bool, optional
            Recompute the shift even if it's already computed
        
        Returns
        -------
        Tuple[float, float]
            (shift_x, shift_y) in pixels at image scale.
        
        Raises
        ------
        ValueError
            If the star cutout has not been extracted.
        """
        star = self.stars[row_index]

        if star.shift is not None and force_recomputation == False:
            return star.shift
        
        if star is None:
            raise ValueError(f"Star {row_index} has not been extracted.")
        
        scale_factor = star.scale_factor
        
        if centroid_radius is None:
            centroid_radius = int(self.DEFAULT_CENTROID_RADIUS * scale_factor)
        if max_shift is None:
            max_shift = self.DEFAULT_MAX_SHIFT * scale_factor
        
        
        # Define central region for centroid fitting
        c_data_center = ((star.cutout.shape[0] - 1) / 2.0, (star.cutout.shape[1] - 1) / 2.0)
        slice_d = (
            slice(int(c_data_center[0]) - centroid_radius, int(c_data_center[0]) + centroid_radius + 1),
            slice(int(c_data_center[1]) - centroid_radius, int(c_data_center[1]) + centroid_radius + 1)
        )


        data = star.cutout[slice_d]

        # Prepare error array from weight map
        if star.cutout_wht is not None:
            wht = star.cutout_wht[slice_d]
            error = np.empty_like(wht, dtype=float)

            valid = np.isfinite(wht) & (wht > 0)
            error[valid] = np.sqrt(1.0 / wht[valid])
            error[~valid] = np.nan

        else:
            error = None
        
        # Fit 2D Gaussian centroid
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The fit may not have converged")
            
                x0, y0 = centroid_2dg(data, error=error)
        except:
            star.shift = (0,0)
            return (0,0)
        

        # Check for fitting failure
        if x0 is None or y0 is None or not np.isfinite(x0) or not np.isfinite(y0):
            star.shift = (0,0)
            return (0,0)

        shift_x = x0 - centroid_radius
        shift_y = y0 - centroid_radius
        
        # Check if shift exceeds maximum
        if np.abs(shift_x) > max_shift or np.abs(shift_y) > max_shift:
            #star.is_valid = False
            #star.failure_reason = f"Shift exceeds maximum: ({shift_x:.2f}, {shift_y:.2f}) > {max_shift}"
            star.shift = (0,0)
            return (0,0)
        
        # Store shift (note: stored as (y, x) for scipy.ndimage.shift compatibility)
        star.shift = (-shift_y, -shift_x)
        
        return (-shift_y, -shift_x)

   
    # =========================================================================
    # Rotation Computation
    # =========================================================================
    
    def compute_rotation(self,
                         row_index: int,
                         order_fft: int,
                         r_min: float = 0.16,
                         r_max: float = 1,
                         angle_of_one_spike: float = 0.0,
                         rotate_first_by_best_pa: bool = True,
                         show: bool = False,
                         force_recomputation : bool = False,
                         verbose: bool = True) -> float:
        """
        Compute the rotation angle for a star's diffraction spikes.
        
        Uses FFT analysis of the polar-transformed image to detect the orientation
        of diffraction spikes and compute the rotation angle.
        
        Parameters
        ----------
        row_index : int
            Index of the star in the catalogue.
        order_fft : int, optional
            Number of diffraction spikes (FFT harmonic order). Default is 6.
        r_min : int, optional
            Minimum radius in arcsec for polar transformation. Default is 0.16.
        r_max : int, optional
            Maximum radius in arcsec for polar transformation. Default is 1.
        angle_of_one_spike : float, optional
            Expected angle of one spike in degrees. Default is 0.0.
        rotate_first_by_best_pa : bool, optional
            If true the angle is computed after rotating the star by -best_pa of the field
        show : bool, optional
            If True, display diagnostic plots. Default is False.
        force_recomputation : bool, optional
            Recompute the shift even if it's already computed
        verbose : bool, optional
            If True, warn the user if background subtraction or shift are not computed. Default is True.
        
        Returns
        -------
        float or None
            Rotation angle in degrees, or None if detection failed.
        """
        
        star = self.stars[row_index]

        if star.rotation is not None and force_recomputation == False:
            return star.rotation
        
        if star is None:
            raise ValueError(f"Star {row_index} has not been extracted.")
        
        if verbose and star.background is None:
            print(f"Warning: no background computed for star {row_index}")
        
        # Use the cutout directly (background should already be subtracted)
        if rotate_first_by_best_pa:
            psf = process_cutout(star.cutout,star.shift,-star.best_pa,order=1)

        elif star.shift is not None:
            psf = process_cutout(star.cutout,star.shift,0.0,order=1)
        
        else:
            if verbose:
                print(f"Warning: no shift computed for star {row_index}")
            psf = star.cutout

        r_min_px = int(round(r_min / self.final_resolution * star.scale_factor))
        r_max_px = int(round(r_max / self.final_resolution * star.scale_factor))

        # Compute rotation angle
        rotation = self._get_rotation_angle(
            psf,
            order_fft=order_fft,
            r_min_px=r_min_px,
            r_max_px=r_max_px,
            angle_of_one_spike=angle_of_one_spike,
            scale_factor = star.scale_factor,
            show=show
        )
        
        star.rotation = rotation

        return rotation
    
    def _get_rotation_angle(self,
                            psf: np.ndarray,
                            order_fft: int = 6,
                            r_min_px: int = 4,
                            r_max_px: int = 12,
                            angle_of_one_spike: float = 0.0,
                            scale_factor: int = 1,
                            show: bool = False) -> float:
        """
        Calculate rotation angle by isolating FFT harmonics and fitting the peak.
        
        This method converts the PSF to polar coordinates, applies FFT filtering
        to isolate the diffraction spike pattern, and uses quadratic interpolation
        to find the sub-pixel peak position.
        
        Parameters
        ----------
        psf : np.ndarray
            Input PSF image (2D array).
        order_fft : int, optional
            Order of the FFT harmonic, corresponding to the number of diffraction
            spikes. Default is 6.
        r_min_px : int, optional
            Minimum radius in pixels for polar transformation. Default is 4.
        r_max_px: int, optional
            Maximum radius in pixels for polar transformation. Default is 12.
        angle_of_one_spike : float, optional
            Expected angle of one spike in degrees, used as reference. Default is 0.0.
        scale_factor: int, optional
            Scale factor of the image. Only used to obtain a uniform imshow scale. Default is 1
        show : bool, optional
            If True, display diagnostic plots. Default is False.
        
        Returns
        -------
        float or None
            Rotation angle in degrees relative to the expected spike position.
        """
        # Step 1: Normalise PSF and convert to polar coordinates
        psf_norm = psf / np.nansum(psf)
        polar_psf = self._convert_to_polar_image(
            psf_norm, 
            r_min_px=r_min_px, 
            r_max_px=r_max_px, 
            order=1
        )
        
        # Display polar conversion if requested
        if show:
            self._plot_polar_conversion(psf_norm, polar_psf, r_min_px, r_max_px, scale_factor= scale_factor)
        
        # Step 2: Compute angular profile by summing over radii
        theta_profile = np.sum(polar_psf, axis=1)
        n_pixels = len(theta_profile)
        
        # Step 3: Apply FFT transformation
        fft_transformation = np.fft.fft(theta_profile)
        
        # Step 4: Harmonic filtering - keep only frequencies that are multiples of order_fft
        # This preserves the spike pattern while removing non-periodic noise
        harmonic_mask = np.zeros_like(fft_transformation, dtype=bool)
        for k in range(1, len(fft_transformation) // 2):
            if k % order_fft == 0:
                harmonic_mask[k] = True      # Positive frequency
                harmonic_mask[-k] = True     # Negative frequency (conjugate)
        
        # Apply harmonic filter
        fft_filtered = fft_transformation.copy()
        fft_filtered[~harmonic_mask] = 0.0
        theta_profile_filtered = np.real(np.fft.ifft(fft_filtered))
        
        # Step 5: Isolate the nth order harmonic only
        nth_order_mask = np.zeros_like(fft_transformation, dtype=bool)
        nth_order_mask[order_fft] = True
        nth_order_mask[-order_fft] = True
        
        fft_nth_order = fft_transformation.copy()
        fft_nth_order[~nth_order_mask] = 0.0
        theta_profile_nth_order = np.real(np.fft.ifft(fft_nth_order))
        
        # Compute zero-mean profile for visualisation
        fft_no_dc = fft_transformation.copy()
        fft_no_dc[0] = 0
        theta_profile_zero_mean = np.real(np.fft.ifft(fft_no_dc))
        
        # Step 6: Exacerbate peaks by combining filtered profiles
        theta_profile_peaks = theta_profile_filtered# * (
        #    theta_profile_nth_order - np.min(theta_profile_nth_order)
        #)
        
        # Step 7: Find peak with sub-pixel accuracy using quadratic interpolation
        idx_max = np.argmax(theta_profile_peaks)
        
        # Get values for quadratic fit (handle wrapping)
        y_center = theta_profile_filtered[idx_max]
        y_left = theta_profile_filtered[idx_max - 1]
        y_right = theta_profile_filtered[(idx_max + 1) % n_pixels]
        
        # Quadratic interpolation: vertex position offset
        denominator = y_left - 2 * y_center + y_right
        if denominator == 0:
            offset = 0.0
        else:
            offset = 0.5 * (y_left - y_right) / denominator
        
        best_pixel_sub = idx_max + offset
        
        # Step 8: Convert pixel coordinate to degrees
        angle_per_pixel = 360.0 / n_pixels
        peak_angle_deg = (best_pixel_sub * angle_per_pixel) % 360
        
        # Compute all peak positions for visualisation
        peaks = (peak_angle_deg + np.linspace(0, 360, order_fft + 1)) % 360
        
        # Step 9: Calculate rotation relative to expected spike position
        # The spikes repeat every (360 / order_fft) degrees
        period = 360.0 / order_fft
        
        # Find the remainder closest to 0 (can be negative)
        rotation = (peak_angle_deg - angle_of_one_spike + period / 2) % period - period / 2
        
        # Display diagnostic plots if requested
        if show:
            self._plot_rotation_diagnostic(
                theta_profile_zero_mean,
                theta_profile_filtered,
                theta_profile_nth_order,
                theta_profile_peaks,
                peaks,
                rotation,
                angle_of_one_spike,
                order_fft,
                n_pixels
            )
        
        
        return rotation
    
    def _convert_to_polar_image(self,
                                 image: np.ndarray,
                                 r_min_px: int = 0,
                                 r_max_px: int = None,
                                 order: int = 1) -> np.ndarray:
        """
        Convert a 2D image to polar coordinates.
        
        Parameters
        ----------
        image : np.ndarray
            Input 2D image.
        r_min_px : int, optional
            Minimum radius in pixels. Default is 0.
        r_max_px : int, optional
            Maximum radius in pixels. Default is half the image size.
        order : int, optional
            Interpolation order for coordinate mapping. Default is 1.
        
        Returns
        -------
        np.ndarray
            Polar image with shape (n_angles, n_radii).
        """
        
        # Image center
        center_y = (image.shape[0] - 1) / 2.0
        center_x = (image.shape[1] - 1) / 2.0
        
        if r_max_px is None:
            r_max_px = int(min(center_y, center_x))
        
        # Define polar grid
        n_radii = r_max_px - r_min_px

        #angle so that each different theta will have different pixels surely
        n_angles = int(2*np.pi//np.arctan(1/r_max_px))
        
        radii = np.linspace(r_min_px, r_max_px, n_radii)
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        
        # Create meshgrid
        R, Theta = np.meshgrid(radii, angles)
        
        # Convert to Cartesian coordinates
        X = center_x + R * np.cos(Theta)
        Y = center_y + R * np.sin(Theta)
        
        # Sample the image at polar coordinates
        polar_image = map_coordinates(
            image, 
            [Y, X], 
            order=order, 
            mode='constant', 
            cval=0.0
        )
        
        return polar_image
    
    def _plot_polar_conversion(self,
                                psf_norm: np.ndarray,
                                polar_psf: np.ndarray,
                                r_min: int,
                                r_max: int,
                                scale_factor: int = 1,
                                data_norm_scale = 1.e-5) -> None:
        """
        Display diagnostic plot for polar coordinate conversion.
        
        Parameters
        ----------
        psf_norm : np.ndarray
            Normalised PSF image.
        polar_psf : np.ndarray
            Polar-transformed PSF.
        r_min : int
            Minimum radius used.
        r_max : int
            Maximum radius used.
        scale_factor : int
            Scale factor if the image used
        """

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Original PSF
        im = ax1.imshow(psf_norm, origin="lower", norm=AsinhNorm(data_norm_scale/scale_factor**2, 0), cmap="magma")
        ax1.set_xlabel("x [px]")
        ax1.set_ylabel("y [px]")
        ax1.set_title("PSF")

        center = ((psf_norm.shape[1] - 1) / 2, (psf_norm.shape[0] - 1) / 2)

        ax1.add_patch(Circle(center, r_max, fill=False, color="w", lw=0.7, ls="--", alpha=0.5))
        ax1.add_patch(Circle(center, r_min, fill=False, color="w", lw=0.7, ls="--", alpha=0.5, label = "r_min, r_max"))

        ax1.legend(labelcolor="w",frameon=False)
        plt.colorbar(im)
        

        # Polar representation
        im = ax2.imshow(polar_psf, origin="lower", norm=AsinhNorm(data_norm_scale/scale_factor**2, 0), 
                   aspect="auto", cmap="magma")
        ax2.set_xlabel("r [px]")
        ax2.set_ylabel(r"$\theta$ [°]")
        ax2.set_yticks(
            np.linspace(0, polar_psf.shape[0], 5), 
            [0, 90, 180, 270, 360]
        )
        ax2.set_xticks(
            np.linspace(0, polar_psf.shape[1] - 1, 5),
            r_min + np.linspace(0, r_max - r_min, 5)
        )
        ax2.set_title("Polar Transform")
        
        plt.colorbar(im)
        plt.tight_layout()
        plt.show()
    
    def _plot_rotation_diagnostic(self,
                                   theta_profile_zero_mean: np.ndarray,
                                   theta_profile_filtered: np.ndarray,
                                   theta_profile_nth_order: np.ndarray,
                                   theta_profile_peaks: np.ndarray,
                                   peaks: np.ndarray,
                                   rotation: float,
                                   angle_of_one_spike: float,
                                   order_fft: int,
                                   n_pixels: int) -> None:
        """
        Display diagnostic plots for rotation angle detection.
        
        Parameters
        ----------
        theta_profile_zero_mean : np.ndarray
            Angular profile with DC component removed.
        theta_profile_filtered : np.ndarray
            Harmonic-filtered angular profile.
        theta_profile_nth_order : np.ndarray
            Nth-order harmonic only.
        theta_profile_peaks : np.ndarray
            Peak-exacerbated profile.
        peaks : np.ndarray
            Detected peak positions in degrees.
        rotation : float
            Computed rotation angle.
        angle_of_one_spike : float
            Reference spike angle.
        order_fft : int
            FFT order (number of spikes).
        n_pixels : int
            Number of angular bins.
        """
        
        theta_rad = np.linspace(0, 2 * np.pi, n_pixels)
        
        # Plot 1: Polar representation of angular profiles
        fig1 = plt.figure(figsize=(4, 4))
        ax_polar = fig1.add_subplot(111, projection="polar")
        
        # Normalise for display
        norm_factor = np.max(np.abs(theta_profile_zero_mean))
        
        ax_polar.plot(
            theta_rad, 
            theta_profile_zero_mean / norm_factor, 
            label="Zero-mean profile", 
            alpha=0.5
        )
        ax_polar.plot(
            theta_rad, 
            theta_profile_filtered / norm_factor, 
            label="Harmonic filtered", 
            color='r'
        )
        ax_polar.plot(
            theta_rad, 
            theta_profile_nth_order / norm_factor, 
            label=f"{order_fft}th order", 
            color='g'
        )
        ax_polar.vlines(
            np.radians(peaks), 
            0, 1, 
            colors='orange', 
            label="Detected peaks"
        )
        ax_polar.vlines(
            np.radians(angle_of_one_spike)+np.linspace(0,2*np.pi,order_fft+1), 
            -0.5, 0, 
            colors='purple', 
            label="Input spike angle"
        )

        ax_polar.axhline(0, color="k", zorder=-1, lw=0.3)
        ax_polar.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax_polar.tick_params(axis='y', labelsize=7)
        ax_polar.set_ylim(-1, 1)
        
        # Position legend outside the polar plot
        angle = np.deg2rad(67.5)
        ax_polar.legend(
            loc="lower left",
            bbox_to_anchor=(0.5 + np.cos(angle) / 2, 0.5 + np.sin(angle) / 2),
            frameon=True,
            fontsize=8
        )
        
        plt.tight_layout()
        plt.show()

    def mask_star_bad_rotation(self, 
                            index_star: int, 
                            angle_tolerance: float = 2.0) -> bool:
        """
        Mask a star if its rotation angle exceeds the tolerance.
        
        Parameters
        ----------
        index_star : int
            Index of the star in the catalogue.
        angle_tolerance : float, optional
            Maximum allowed rotation angle in degrees. Stars with 
            |rotation| > angle_tolerance are masked. Default is 2.0.
        
        Returns
        -------
        bool
            True if the star was masked, False otherwise.
        
        """

        star = self.stars[index_star]
        
        if star is None:
            raise ValueError(f"Star {index_star} has not been extracted.")
        
        if star.rotation is None:
            raise ValueError(f"Star {index_star} has no computed rotation angle.")
        
        if np.abs(star.rotation) > angle_tolerance:
            star.mask = True
            star.failure_reason = f"Bad rotation: {star.rotation:.2f}°"
            return True
        
        return False
    
    def check_aperture_ratio(self, 
                            index_star: int, 
                            apertures: Tuple[float, float] | np.ndarray = (0.16, 0.32),
                            sigma_threshold: float = 3.0,
                            mask_high_values: bool = False) -> bool:
        """
        Mask a star if its aperture flux ratio deviates significantly from the expected value.
        
        Stars with anomalous aperture ratios may indicate:
        - Saturation or non-linearity effects (ratio too high)
        - Contamination from nearby sources (ratio too low)
        - Extended or resolved sources rather than point sources
        - PSF fitting issues
        
        Parameters
        ----------
        index_star : int
            Index of the star in the catalogue.
        apertures : np.ndarray, optional
            Aperture diameters in arcseconds. Default is [0.16, 0.32].
        sigma_threshold : float, optional
            Number of standard deviations from the median ratio beyond which 
            a star is considered bad. Default is 3.0.
        mask_high_values : bool, optional
            If True, also mask stars with ratios significantly higher than the median. If False, only mask stars with 
            ratios lower than expected. Default is False.
        
        Returns
        -------
        bool
            True if the star was masked, False if it passed the check.
        
        """
        star = self.stars[index_star]
        
        if star is None:
            raise ValueError(f"Star {index_star} has not been extracted.")
        
        if star.cutout_elaborated is None:
            raise ValueError(f"Star {index_star} is not aligned. Call align_and_rotate_star first.")

        img_data = star.cutout_elaborated

        center = np.array([(img_data.shape[0] - 1) / 2.0, (img_data.shape[1] - 1) / 2.0])
    
        # Convert aperture diameters to radii in pixels
        r_small = apertures[0] / (2 * self.final_resolution)
        r_large = apertures[1] / (2 * self.final_resolution)
        
        ap_small = CircularAperture(center, r=r_small)
        ap_large = CircularAperture(center, r=r_large)
        
        flux_small = ap_small.do_photometry(img_data)[0][0]
        flux_large = ap_large.do_photometry(img_data)[0][0]
        

        ap_ratio =  flux_small / flux_large
        
        # Calculate deviation from median in units of sigma
        deviation = ap_ratio - self.median_ratio_apertures
        threshold = sigma_threshold * self.std_ratio_apertures
        
        # Check if ratio is too low 
        is_too_low = deviation < -threshold
        
        # Check if ratio is too high 
        is_too_high = mask_high_values and (deviation > threshold)
        
        if is_too_low or is_too_high:
            star.is_valid = False
            
            if is_too_low:
                reason = "too low"
            else:
                reason = "too high"
            
            star.failure_reason = f"Aperture ratio {ap_ratio:.4f} is {reason}"
            return True
        
        return False
    
    def align_and_rotate_star(self,
                            index_star: int,
                            final_angle: float = 0.0,
                            fine_rotation: bool = False,
                            order: int = 3) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Align and rotate a star cutout (wrapper for external function)."""
        star = self.stars[index_star]
        
        if star is None:
            raise ValueError(f"Star {index_star} has not been extracted.")
        
        if star.shift is None:
            raise ValueError(f"Star {index_star} has no shift. Call compute_shift first.")
        
        reduced_image, reduced_wht = align_and_rotate_star(
            cutout=star.cutout,
            cutout_wht=star.cutout_wht,
            shift=star.shift,
            best_pa=star.best_pa,
            scale_factor=star.scale_factor,
            upsample_factor=self.upsample_factor,
            final_size=self.final_size,
            final_angle=final_angle,
            rotation=star.rotation,
            fine_rotation=fine_rotation,
            order=order
        )
        
        star.cutout_elaborated = reduced_image
        star.cutout_wht_elaborated = reduced_wht
        
        return reduced_image, reduced_wht


    def normalize_star(self,
                    index_star: int,
                    norm_radius: float = 1.0) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Normalize a star cutout by its flux within a circular aperture."""
        star = self.stars[index_star]
        
        if star is None:
            raise ValueError(f"Star {index_star} has not been extracted.")
        
        if star.cutout_elaborated is None:
            raise ValueError(
                f"Star {index_star} is not aligned. Call align_and_rotate_star first."
            )
        
        # Usa la funzione esterna
        norm_data, norm_wht = normalize_cutout(
            cutout=star.cutout_elaborated,
            cutout_wht=star.cutout_wht_elaborated,
            norm_radius=norm_radius,
            final_resolution=self.final_resolution
        )
        
        star.cutout_elaborated = norm_data
        star.cutout_wht_elaborated = norm_wht
        
        return norm_data, norm_wht
        
    # =========================================================================
    # Utility Methods
    # =========================================================================
    def unknown_stars(self) -> list:
        """
        Get list of indices of stars thta are not good or bad.
        
        Returns
        -------
        list
            Indices of stars.
        """
        return [i for i, star in enumerate(self.stars) if star is not None and star.is_valid==-1 and not star.mask]

    def get_valid_stars(self) -> list:
        """
        Get list of indices of valid stars.
        
        Returns
        -------
        list
            Indices of stars that passed all quality checks.
        """
        return [i for i, star in enumerate(self.stars) if star is not None and star.is_valid==1 and not star.mask]
    
    def get_invalid_stars(self) -> list:
        """
        Get list of indices of invalid and masked stars with their failure reasons.
        
        Returns
        -------
        list
            List of (index, reason) tuples.
        """
        invalid = []
        for i, star in enumerate(self.stars):
            if star is None:
                invalid.append((i, "Not extracted"))
            elif star.is_valid==0 or star.mask:
                invalid.append((i, star.failure_reason or "Unknown"))
        return invalid
    
    def summary(self) -> None:
        """Print a summary of the star processing status."""
        n_total = len(self.stars)
        n_extracted = sum(1 for s in self.stars if s is not None)
        n_valid = sum(1 for s in self.stars if s is not None and s.is_valid==1)
        n_unknown = sum(1 for s in self.stars if s is not None and s.is_valid==-1)
        n_with_shift = sum(1 for s in self.stars if s is not None and s.shift is not None)
        n_with_rotation = sum(1 for s in self.stars if s is not None and s.rotation is not None)
        
        print(f"StarCutouts Summary:")
        print(f"  Total stars in catalogue: {n_total}")
        print(f"  Cutouts extracted: {n_extracted} ({100*n_extracted/n_total:.1f}%)")
        print(f"  Valid stars: {n_valid} ({100*n_valid/n_total:.1f}%)")
        print(f"  Not yet checked stars: {n_unknown} ({100*n_unknown/n_total:.1f}%)")
        print(f"  Stars with shift computed: {n_with_shift}")
        print(f"  Stars with rotation computed: {n_with_rotation}")