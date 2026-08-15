from dataclasses import dataclass
from typing import Optional, List, Literal, Tuple, Dict
import numpy as np
from scipy.optimize import minimize
from photutils.centroids import centroid_2dg
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm

from .core import StarCutouts, process_cutout, align_and_rotate_star, normalize_cutout


@dataclass
class RefinementResult:
    """Results from iterative PSF refinement."""
    n_iterations: int
    shift_corrections: List[np.ndarray]
    total_shifts: List[np.ndarray]
    rms_correction_per_iteration: np.ndarray
    epsf_fwhm_per_iteration: np.ndarray
    epsf_history: List[np.ndarray]
    final_epsf: np.ndarray
    converged: bool
    valid_indices: List[int]


class EPIBuilder:
    """
    Builds effective PSF (ePSF) through iterative shift refinement and stacking.
    
    IMPORTANT: This class does NOT modify the original cutouts. All shifts and
    elaborated cutouts are stored internally.
    """
    
    DEFAULT_MAX_SHIFT = 5.0 #px
    DEFAULT_FIT_ORDER = 5
    DEFAULT_SHIFT_ORDER = 3
    DEFAULT_K_SIGMA = 3.0
    DEFAULT_MAXITERS = 10
    DEFAULT_CENTROID_RADIUS = 7 #px
    DEFAULT_NORM_RADIUS = 1.0  #arcsec

    def __init__(self, cutouts: StarCutouts, norm_radius: float = DEFAULT_NORM_RADIUS):
        self.cutouts = cutouts
        self.norm_radius = norm_radius  # NUOVO
        self.current_epsf: Optional[np.ndarray] = None
        self._last_refinement: Optional[RefinementResult] = None
        
        # Internal storage (does NOT modify original cutouts)
        self._shifts: Dict[int, Tuple[float, float]] = {}
        self._elaborated: Dict[int, np.ndarray] = {}
        self._elaborated_wht: Dict[int, np.ndarray] = {}
        
        # Initialize with original values
        self._initialize_from_cutouts()
    
    def _initialize_from_cutouts(self) -> None:
        """Copy initial shifts and elaborated cutouts from original cutouts."""
        self._shifts.clear()
        self._elaborated.clear()
        self._elaborated_wht.clear()
        
        for i, star in enumerate(self.cutouts.stars):
            if star is not None and star.is_valid == 1 and not star.mask:
                # Copy shift
                if star.shift is not None:
                    self._shifts[i] = star.shift
                
                # Copy elaborated cutouts
                if star.cutout_elaborated is not None:
                    self._elaborated[i] = star.cutout_elaborated.copy()
                if star.cutout_wht_elaborated is not None:
                    self._elaborated_wht[i] = star.cutout_wht_elaborated.copy()
    
    def reset(self) -> None:
        """Reset internal storage to original cutout values."""
        self._initialize_from_cutouts()
        self.current_epsf = None
        self._last_refinement = None
    
    # =========================================================================
    # VALID STAR UTILITIES
    # =========================================================================
    
    def get_valid_indices(self) -> List[int]:
        """Get indices of valid stars for stacking."""
        return [
            i for i, star in enumerate(self.cutouts.stars)
            if star is not None 
            and star.is_valid == 1
            and not star.mask
            and (i in self._elaborated or star.cutout_elaborated is not None)
        ]
    
    def get_n_valid(self) -> int:
        """Return number of valid stars."""
        return len(self.get_valid_indices())
    
    def _get_shift(self, star_index: int) -> Tuple[float, float]:
        """Get shift from internal storage or fall back to original."""
        if star_index in self._shifts:
            return self._shifts[star_index]
        star = self.cutouts.stars[star_index]
        if star is not None and star.shift is not None:
            return star.shift
        return (0.0, 0.0)
    
    def _get_stacking_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Collect cutouts and weights from valid stars (uses internal storage)."""
        valid_indices = self.get_valid_indices()
        
        cutouts_list = []
        weights_list = []
        
        for idx in valid_indices:
            # Always prefer internal storage
            if idx in self._elaborated:
                cutouts_list.append(self._elaborated[idx])
                if idx in self._elaborated_wht:
                    weights_list.append(self._elaborated_wht[idx])
                else:
                    weights_list.append(np.ones_like(self._elaborated[idx]))
            else:
                # Fallback to original (should not happen after init)
                star = self.cutouts.stars[idx]
                cutouts_list.append(star.cutout_elaborated)
                if star.cutout_wht_elaborated is not None:
                    weights_list.append(star.cutout_wht_elaborated)
                else:
                    weights_list.append(np.ones_like(star.cutout_elaborated))
        
        return np.array(cutouts_list), np.array(weights_list)
    
    def _align_star_internal(self, 
                            star_index: int, 
                            shift: Tuple[float, float], 
                            final_angle: float = 0.0, 
                            fine_rotation: bool = False,
                            order: int = 3,
                            normalize: bool = True,
                            norm_radius: float = 1.0) -> None:

        """Align star and store result internally (does NOT modify original)."""
        star = self.cutouts.stars[star_index]

        best_pa = star.best_pa if star.best_pa is not None else 0
        
        reduced_image, reduced_wht = align_and_rotate_star(
            cutout=star.cutout,
            cutout_wht=star.cutout_wht,
            cutout_mask=star.cutout_mask,
            shift=shift,
            best_pa=best_pa,
            scale_factor=star.scale_factor,
            upsample_factor=self.cutouts.upsample_factor,
            final_size=self.cutouts.cutout_shape,
            final_angle=final_angle,
            rotation=star.rotation,
            fine_rotation=fine_rotation,
            order=order
        )
        
        # Normalize if requested
        if normalize:
            reduced_image, reduced_wht = normalize_cutout(
                cutout=reduced_image,
                cutout_wht=reduced_wht,
                norm_radius=norm_radius,
                final_resolution=self.cutouts.final_resolution
            )
        
        self._elaborated[star_index] = reduced_image
        if reduced_wht is not None:
            self._elaborated_wht[star_index] = reduced_wht
    
    # =========================================================================
    # ePSF METRICS
    # =========================================================================
    
    def compute_fwhm(self, epsf: np.ndarray, method: str = 'half_max', 
                 n_radial_samples: int = 500, n_angular_samples: int = 360) -> float:
        """
        Compute FWHM of the ePSF at subpixel resolution.
        
        Parameters
        ----------
        epsf : np.ndarray
            2D ePSF image.
        method : str
            Method to use: 'half_max' (interpolated) or 'gaussian' (fit).
        n_radial_samples : int
            Number of radial samples for interpolation. Default is 500.
        n_angular_samples : int
            Number of angular samples for azimuthal averaging. Default is 360.
        
        Returns
        -------
        float
            FWHM in pixels.
        """
        from scipy.ndimage import map_coordinates
        from scipy.interpolate import UnivariateSpline
        from scipy.optimize import brentq
        
        center = (np.array(epsf.shape) - 1) / 2.0
        r_max = min(center) * 0.95  # Stay slightly inside image
        
        # Create fine radial grid
        r_fine = np.linspace(0, r_max, n_radial_samples)
        theta = np.linspace(0, 2 * np.pi, n_angular_samples, endpoint=False)
        
        # Sample image at subpixel positions using bilinear interpolation
        radial_profile = np.zeros(n_radial_samples)
        
        for i, r in enumerate(r_fine):
            if r == 0:
                # At center, just take the center pixel value (interpolated)
                radial_profile[i] = map_coordinates(epsf, [[center[0]], [center[1]]], 
                                                    order=3, mode='constant')[0]
            else:
                # Sample at n_angular_samples points around the circle
                y_coords = center[0] + r * np.sin(theta)
                x_coords = center[1] + r * np.cos(theta)
                
                # Interpolate values at these coordinates
                values = map_coordinates(epsf, [y_coords, x_coords], 
                                        order=3, mode='constant')
                radial_profile[i] = np.mean(values)
        
        if method == 'half_max':
            peak = np.max(radial_profile)
            half_max = peak / 2
            
            # Check if profile ever drops below half_max
            if np.all(radial_profile > half_max):
                return r_max * 2
            
            # Create smooth spline interpolation
            # Use smoothing to handle noise
            spline = UnivariateSpline(r_fine, radial_profile, s=0, k=3)
            
            # Find where profile crosses half_max
            # First, find approximate location
            above_half = radial_profile > half_max
            if not np.any(~above_half):
                return r_fine[-1] * 2
            
            first_below_idx = np.argmax(~above_half)
            if first_below_idx == 0:
                return r_fine[0] * 2
            
            # Bracket the root
            r_low = r_fine[first_below_idx - 1]
            r_high = r_fine[first_below_idx]
            
            # Extend bracket slightly if needed
            while spline(r_low) < half_max and first_below_idx > 1:
                first_below_idx -= 1
                r_low = r_fine[first_below_idx - 1]
            
            while spline(r_high) > half_max and first_below_idx < len(r_fine) - 1:
                first_below_idx += 1
                r_high = r_fine[first_below_idx]
            
            # Use Brent's method to find exact crossing point
            try:
                r_half = brentq(lambda r: spline(r) - half_max, r_low, r_high)
            except ValueError:
                # Fallback to linear interpolation if brentq fails
                p1, p2 = radial_profile[first_below_idx - 1], radial_profile[first_below_idx]
                r1, r2 = r_fine[first_below_idx - 1], r_fine[first_below_idx]
                if p1 != p2:
                    r_half = r1 + (half_max - p1) * (r2 - r1) / (p2 - p1)
                else:
                    r_half = (r1 + r2) / 2
            
            return 2 * r_half
        
        else:  # gaussian fit
            from scipy.optimize import curve_fit
            
            def gaussian(r, amp, sigma, offset):
                return amp * np.exp(-r**2 / (2 * sigma**2)) + offset
            
            try:
                p0 = [np.max(radial_profile) - np.min(radial_profile), 
                    2.0, 
                    np.min(radial_profile)]
                popt, _ = curve_fit(gaussian, r_fine, radial_profile, p0=p0, maxfev=2000)
                sigma = abs(popt[1])
                return 2.355 * sigma  # FWHM = 2*sqrt(2*ln(2)) * sigma
            except:
                # Fallback to half_max method
                return self.compute_fwhm(epsf, method='half_max')
    
    def compute_encircled_energy(self, 
                                 epsf: np.ndarray, 
                                 fractions: List[float] = [0.5, 0.8, 0.9]) -> dict:
        """Compute encircled energy radii."""
        center = (np.array(epsf.shape) - 1) / 2.0
        y, x = np.ogrid[:epsf.shape[0], :epsf.shape[1]]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        r_flat = r.flatten()
        epsf_flat = epsf.flatten()
        sort_idx = np.argsort(r_flat)
        r_sorted = r_flat[sort_idx]
        epsf_sorted = epsf_flat[sort_idx]
        
        cumsum = np.cumsum(epsf_sorted)
        total = cumsum[-1]
        
        results = {}
        for frac in fractions:
            target = frac * total
            idx = np.searchsorted(cumsum, target)
            results[frac] = r_sorted[min(idx, len(r_sorted)-1)]
        
        return results
    
    # =========================================================================
    # SIGMA CLIPPING
    # =========================================================================
    
    def _mad_sigma_clip(self, data: np.ndarray, sigma: float, 
                        maxiters: int, axis: int = 0) -> np.ndarray:
        """MAD-based sigma clipping."""

        mask = data==0
        
        for _ in range(maxiters):
            data_masked = np.ma.array(data,mask=mask)
            reference = np.ma.median(data_masked, axis=axis, keepdims=True)
            distance = np.abs(data - reference)
            
            masked_distance = np.ma.array(distance, mask=mask)
            mad = np.ma.median(masked_distance, axis=axis, keepdims=True).filled(1.0)
            
            new_mask = distance > sigma * mad
            
            if np.array_equal(new_mask, mask):
                break
            mask = new_mask
        
        return mask
    
    def _weighted_sigma_clip(self, data: np.ndarray, weights: np.ndarray,
                             sigma: float, maxiters: int, 
                             use_median: bool = False, axis: int = 0) -> np.ndarray:
        """Weighted sigma clipping."""
        mask = np.zeros(data.shape, dtype=bool)
        
        for _ in range(maxiters):
            w = np.where(mask, 0.0, weights)
            w_sum = np.sum(w, axis=axis, keepdims=True)
            w_sum = np.where(w_sum == 0, 1.0, w_sum)
            
            if use_median:
                reference = np.median(data, axis=axis, keepdims=True)
            else:
                reference = np.sum(data * w, axis=axis, keepdims=True) / w_sum
            
            masked_data = np.ma.array(data, mask=mask)
            std = np.ma.std(masked_data, axis=axis, keepdims=True).filled(1.0)
            
            new_mask = np.abs(data - reference) > sigma * std
            
            if np.array_equal(new_mask, mask):
                break
            mask = new_mask
        
        return mask
    
    # =========================================================================
    # STACKING
    # =========================================================================
    
    def stack(self,
              rejection_mode: Literal['none', 'weighted', 'mad'] = 'mad',
              k_sigma: float = DEFAULT_K_SIGMA,
              maxiters: int = DEFAULT_MAXITERS,
              verbose: bool = True) -> np.ndarray:
        """Stack valid star cutouts with optional sigma rejection."""
        n_valid = self.get_n_valid()
        
        if n_valid == 0:
            raise ValueError("No valid stars available for stacking")
        
        data, weights = self._get_stacking_data()
        
        if verbose:
            print(f"Stacking {n_valid} stars | rejection='{rejection_mode}' | k_sigma={k_sigma}")
        
        if rejection_mode == 'none':
            mask = np.zeros(data.shape, dtype=bool)
        elif rejection_mode == 'mad':
            mask = self._mad_sigma_clip(data, k_sigma, maxiters)
        elif rejection_mode == 'weighted':
            use_median = n_valid < 10
            mask = self._weighted_sigma_clip(data, weights, k_sigma, maxiters, use_median)
        else:
            raise ValueError(f"Unknown rejection_mode: '{rejection_mode}'")
        
        weights_masked = weights.copy()
        weights_masked[mask] = 0.0
        
        if verbose:
            rejected = np.sum(mask)
            total = np.prod(data.shape)
            print(f"  Rejected: {rejected}/{total} pixels ({100*rejected/total:.2f}%)")
        
        numerator = np.sum(data * weights_masked, axis=0)
        denominator = np.sum(weights_masked, axis=0)
        denominator[denominator == 0] = 1.0
        
        self.current_epsf = numerator / denominator
        
        return self.current_epsf
    
    # =========================================================================
    # SHIFT FITTING
    # =========================================================================
    
    def _fit_shift_to_original(self,
                               star_index: int,
                               epsf: np.ndarray,
                               centroid_radius: int = None,
                               max_shift: float = DEFAULT_MAX_SHIFT,
                               fit_order: int = DEFAULT_FIT_ORDER) -> Tuple[float, float]:
        """Fit TOTAL shift needed to align original cutout to ePSF."""
        star = self.cutouts.stars[star_index]
        
        if star is None or star.cutout is None:
            return 0.0, 0.0
        
        scale_factor = star.scale_factor
        
        if centroid_radius is None:
            centroid_radius = int(self.DEFAULT_CENTROID_RADIUS * scale_factor)
        
        output_shape = star.cutout.shape
        
        rot_angle = star.best_pa if star.best_pa is not None else 0
        if star.rotation is not None:
            rot_angle -= star.rotation
        
        probe_psf = process_cutout(
            data=epsf,
            shift=(0, 0),
            angle=rot_angle,
            scale_factor=scale_factor,
            output_shape=output_shape,
            order=fit_order
        )
        
        center = (np.array(star.cutout.shape) - 1) / 2.0
        c_int = center.astype(int)
        
        slice_fit = (
            slice(c_int[0] - centroid_radius, c_int[0] + centroid_radius + 1),
            slice(c_int[1] - centroid_radius, c_int[1] + centroid_radius + 1)
        )
        
        data = star.cutout[slice_fit]
        psf_crop = probe_psf[slice_fit]

        mask_data = star.cutout_mask[slice_fit]
        
        if star.cutout_wht is not None:
            wht = star.cutout_wht[slice_fit].copy()
            wht[~np.isfinite(wht)] = 0.0
            wht[wht < 0] = 0.0
        else:
            wht = np.ones_like(data)

        wht[mask_data] = 0
        
        wht_sum = np.sum(wht)
        wht_norm = wht / wht_sum if wht_sum > 0 else np.ones_like(wht) / wht.size
        
        def residual(params):
            shift_x, shift_y, amplitude, offset = params
            shifted_psf = process_cutout(
                data=psf_crop,
                shift=(shift_y, shift_x),
                angle=0.0,
                scale_factor=1,
                output_shape=psf_crop.shape,
                order=fit_order
            )
            model = amplitude * shifted_psf + offset
            return np.sum(wht_norm * (data - model)**2)
        
        p0 = [0.0, 0.0, np.max(data) / (np.max(psf_crop) + 1e-10), np.median(data)]
        bounds = [(-max_shift, max_shift), (-max_shift, max_shift), (0, None), (None, None)]
        
        result = minimize(residual, p0, method='Powell', bounds=bounds)
        
        if result.success:
            shift_x, shift_y = result.x[0], result.x[1]
        else:

            if star.cutout_wht is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    error = np.sqrt(1.0 / (wht + 1e-50))
                    error[~np.isfinite(error)] = np.nanmax(error[np.isfinite(error)])
            else:
                error = None
            
            cx, cy = centroid_2dg(data, error=error, mask=mask_data)
            shift_x = cx - centroid_radius
            shift_y = cy - centroid_radius
        
        return -shift_y, -shift_x
    
    # =========================================================================
    # ITERATIVE REFINEMENT
    # =========================================================================
    def _recenter_epsf(self, epsf: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        """
        Recenter ePSF to geometric center using centroid.
        
        Returns
        -------
        recentered_epsf : np.ndarray
            ePSF shifted to center.
        centroid_offset : Tuple[float, float]
            (offset_y, offset_x) that was applied.
        """
    
        center = (np.array(epsf.shape) - 1) / 2.0
        
        # Find centroid using center of mass or 2D Gaussian
        # Use weighted centroid for robustness
        y, x = np.ogrid[:epsf.shape[0], :epsf.shape[1]]
        
        # Mask to use only central region for centroid
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        mask = r < min(center) * 0.5  # Use central 50%
        
        epsf_masked = np.where(mask, epsf, 0)
        total = np.sum(epsf_masked)
        
        if total > 0:
            centroid_y = np.sum(y * epsf_masked) / total
            centroid_x = np.sum(x * epsf_masked) / total
        else:
            centroid_y, centroid_x = center
        
        # Offset from geometric center
        offset_y = centroid_y - center[0]
        offset_x = centroid_x - center[1]
        
        # Shift ePSF to center it
        recentered = process_cutout(
            data=epsf,
            shift=(-offset_y, -offset_x),  # Shift in opposite direction
            angle=0.0,
            scale_factor=1,
            output_shape=epsf.shape,
            order=3
        )
        
        return recentered, (offset_y, offset_x)
    
    def refine_shifts(self,
                  n_iterations: int = 3,
                  rejection_mode: Literal['none', 'weighted', 'mad'] = 'mad',
                  k_sigma: float = DEFAULT_K_SIGMA,
                  maxiters: int = DEFAULT_MAXITERS,
                  max_shift: float = DEFAULT_MAX_SHIFT,
                  fit_order: int = DEFAULT_FIT_ORDER,
                  shift_order: int = DEFAULT_SHIFT_ORDER,
                  convergence_threshold: float = 0.01,
                  recenter_epsf: bool = True,
                  normalize: bool = True, 
                  verbose: bool = True) -> RefinementResult:
        """
        Iteratively refine star shifts by fitting original cutouts to stacked ePSF.
        
        This does NOT modify the original cutouts. All changes are
        stored internally in self._shifts and self._elaborated.
        """
        # Reset internal storage to original values at the start
        self._initialize_from_cutouts()
        
        valid_indices = self.get_valid_indices()
        n_valid = len(valid_indices)
        
        if n_valid == 0:
            raise ValueError("No valid stars for refinement")
        
        if verbose:
            print(f"{'='*60}")
            print(f"Starting iterative shift refinement")
            print(f"  Valid stars: {n_valid}")
            print(f"  Max iterations: {n_iterations}")
            print(f"  Rejection: {rejection_mode}, k_sigma={k_sigma}")
            print(f"  Recenter ePSF: {recenter_epsf}")
            print(f"  Convergence threshold: {convergence_threshold} px")
            print(f"{'='*60}")
        
        # Compute initial ePSF
        epsf_initial = self.stack(rejection_mode=rejection_mode, k_sigma=k_sigma, 
                                maxiters=maxiters, verbose=verbose)
        
        # RICENTRALIZZA L'ePSF INIZIALE
        if recenter_epsf:
            epsf_initial, initial_offset = self._recenter_epsf(epsf_initial)
            self.current_epsf = epsf_initial
            if verbose:
                print(f"  Initial ePSF recentered by: ({initial_offset[0]:.4f}, {initial_offset[1]:.4f}) px")
        
        fwhm_initial = self.compute_fwhm(epsf_initial)
        
        if verbose:
            print(f"  Initial FWHM: {fwhm_initial:.3f} px")
        
        shift_corrections = []
        total_shifts = []
        rms_history = []
        fwhm_history = [fwhm_initial]
        epsf_history = [epsf_initial.copy()]
        epsf_offsets = []  # Track offsets for debugging
        converged = False
        
        for iteration in range(n_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1}/{n_iterations} ---")
            
            epsf = self.current_epsf
            
            # Fit shifts
            corrections = np.zeros((n_valid, 2))
            new_shifts = np.zeros((n_valid, 2))
            
            iterator = tqdm(enumerate(valid_indices), total=n_valid,
                        desc="Fitting shifts", disable=not verbose)
            
            for i, star_idx in iterator:
                old_shift = np.array(self._get_shift(star_idx))
                
                fitted_y, fitted_x = self._fit_shift_to_original(
                    star_idx, epsf,
                    max_shift=max_shift,
                    fit_order=fit_order
                )
                new_shift = np.array([fitted_y, fitted_x])
                
                correction = new_shift - old_shift
                corrections[i] = correction
                new_shifts[i] = new_shift
                
                self._shifts[star_idx] = (fitted_y, fitted_x)
            
            shift_corrections.append(corrections.copy())
            total_shifts.append(new_shifts.copy())
            
            # Update elaborated cutouts
            if verbose:
                print("Updating elaborated cutouts...")
                for star_idx in valid_indices:
                    self._align_star_internal(
                        star_idx, 
                        self._shifts[star_idx],
                        final_angle=0.0,
                        fine_rotation=False,
                        normalize=normalize,
                        order=shift_order,
                        norm_radius=self.norm_radius
                    )
            
            # Stack
            epsf = self.stack(
                rejection_mode=rejection_mode,
                k_sigma=k_sigma,
                maxiters=maxiters,
                verbose=verbose
            )
            
            # center ePSF
            if recenter_epsf:
                epsf, offset = self._recenter_epsf(epsf)
                self.current_epsf = epsf
                epsf_offsets.append(offset)
                if verbose:
                    print(f"  ePSF recentered by: ({offset[0]:.4f}, {offset[1]:.4f}) px")
            
            epsf_history.append(epsf.copy())
            
            fwhm = self.compute_fwhm(epsf)
            fwhm_history.append(fwhm)
            if verbose:
                print(f"  ePSF FWHM: {fwhm:.3f} px")
            
            # Check convergence
            rms_correction = np.sqrt(np.mean(corrections**2))
            rms_history.append(rms_correction)
            
            if verbose:
                print(f"  RMS correction: {rms_correction:.4f} px")
                print(f"  Max correction: {np.max(np.abs(corrections)):.4f} px")
                # Show mean shift to detect drift
                mean_shift = np.mean(new_shifts, axis=0)
                print(f"  Mean shift: ({mean_shift[0]:.4f}, {mean_shift[1]:.4f}) px")
            
            if rms_correction < convergence_threshold:
                if verbose:
                    print(f"\n✓ Converged after {iteration + 1} iterations")
                converged = True
                break
        
        result = RefinementResult(
            n_iterations=len(shift_corrections),
            shift_corrections=shift_corrections,
            total_shifts=total_shifts,
            rms_correction_per_iteration=np.array(rms_history),
            epsf_fwhm_per_iteration=np.array(fwhm_history),
            epsf_history=epsf_history,
            final_epsf=self.current_epsf,
            converged=converged,
            valid_indices=valid_indices
        )
        
        self._last_refinement = result
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Refinement complete: {result.n_iterations} iterations")
            print(f"Converged: {result.converged}")
            final_mean = np.mean(total_shifts[-1], axis=0)
            print(f"{'='*60}")
        
        return result
    
    # =========================================================================
    # MULTI-ANGLE ePSF
    # =========================================================================
    
    def build_multi_angle_epsf(self,
                           angles: Optional[Dict[str, float] | List[float]] = None,
                           output_shape: Optional[Tuple[int, int] | np.ndarray] = None,
                           rejection_mode: Literal['none', 'weighted', 'mad'] = 'mad',
                           k_sigma: float = DEFAULT_K_SIGMA,
                           maxiters: int = DEFAULT_MAXITERS,
                           shift_order: int = DEFAULT_SHIFT_ORDER,
                           fine_rotation: bool = False,
                           normalize: bool = True,
                           verbose: bool = True) -> Dict[str, np.ndarray]:
        """Build ePSF at multiple target angles.

        Parameters
        ----------
        angles : Dict[str, float] or List[float] or None
            Target angles. 
            - If dict: keys are names (e.g., field names), values are angles in degrees
            - If list: creates names as "angle_{a}" for each angle
            - If None: defaults to [0, 45, 90, 135]
        output_shape : Tuple[int, int] or np.ndarray or None
            If specified, crop the final ePSF to this shape (centered).
            If None, keep the original shape.
        rejection_mode : {'none', 'weighted', 'mad'}
            Rejection method for stacking. Default is 'mad'.
        k_sigma : float
            Sigma threshold for rejection. Default is 3.0.
        maxiters : int
            Maximum iterations for sigma clipping. Default is 10.
        fine_rotation : bool
            If True, apply fine rotation correction. Default is False.
        normalize : bool
            If True, normalize each cutout by aperture flux. Default is True.
        verbose : bool
            Print progress. Default is True.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping angle names to ePSF arrays.

        Raises
        ------
        ValueError
            If no valid stars are available or output_shape is larger than original.

        Examples
        --------
        >>> # Build for specific fields with their position angles
        >>> psfs = builder.build_multi_angle_epsf(
        ...     angles={'A2744': 45.2, 'MACS0416': 120.5, 'generic': 0.0},
        ...     output_shape=(101, 101)
        ... )
        >>> 
        >>> # Build for regular angles
        >>> psfs = builder.build_multi_angle_epsf(angles=[0, 30, 60, 90, 120, 150])
        >>> 
        >>> # Save to FITS
        >>> for name, psf in psfs.items():
        ...     fits.PrimaryHDU(data=psf).writeto(f'{name}.fits', overwrite=True)
        """

        if angles is None:
            angles = [0]
        
        # Convert list to dict
        if isinstance(angles, list):
            angles_dict = {f"angle_{int(a)}": float(a) for a in angles}
        else:
            angles_dict = angles
        
        valid_indices = self.get_valid_indices()
        
        if len(valid_indices) == 0:
            raise ValueError("No valid stars for ePSF building")
        
        # Compute slice for cropping (if output_shape specified)
        crop_slice = None
        if output_shape is not None:
            output_shape = np.asarray(output_shape)
            original_shape = np.asarray(self.cutouts.cutout_shape)
            
            if np.any(output_shape > original_shape):
                raise ValueError(
                    f"output_shape {tuple(output_shape)} cannot be larger than "
                    f"original shape {tuple(original_shape)}"
                )
            
            # Compute centered slice
            start = (original_shape - output_shape) // 2
            end = start + output_shape
            crop_slice = (slice(start[0], end[0]), slice(start[1], end[1]))
        
        if verbose:
            print(f"{'='*60}")
            print(f"Building multi-angle ePSF")
            print(f"  Angles: {list(angles_dict.keys())}")
            print(f"  Valid stars: {len(valid_indices)}")
            if output_shape is not None:
                print(f"  Output shape: {tuple(output_shape)}")
            print(f"{'='*60}")
        
        results = {}
        
        for name, target_angle in angles_dict.items():
            if verbose:
                print(f"\n--- {name} (PA = {target_angle}°) ---")
            
            # Align all stars to target angle
            for star_idx in valid_indices:
                shift = self._get_shift(star_idx)
                self._align_star_internal(
                    star_idx,
                    shift,
                    final_angle=target_angle,
                    fine_rotation=fine_rotation,
                    normalize=normalize,
                    order = shift_order,
                    norm_radius=self.norm_radius
                )
            
            # Stack
            epsf = self.stack(
                rejection_mode=rejection_mode,
                k_sigma=k_sigma,
                maxiters=maxiters,
                verbose=verbose
            )
            
            # Crop if requested
            if crop_slice is not None:
                epsf = epsf[crop_slice]
            
            # Normalize the total flux to 1
            epsf = epsf/ np.sum(epsf)
            
            results[name] = epsf.copy()
            
            if verbose:
                fwhm = self.compute_fwhm(epsf)
                print(f"  FWHM: {fwhm:.3f} px")
                print(f"  Shape: {epsf.shape}")
        
        # Store last one as current
        if results:
            self.current_epsf = list(results.values())[-1]
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Generated {len(results)} ePSFs")
            print(f"{'='*60}")
        
        return results
    
    # =========================================================================
    # PLOTTING
    # =========================================================================
    
    def plot_convergence_results(self, figsize: Tuple[int, int] = (12,8)) -> None:
        """Plot convergence diagnostics."""
        if self._last_refinement is None:
            print("No refinement history. Run refine_shifts() first.")
            return
        
        result = self._last_refinement
        n_iter = result.n_iterations
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: RMS correction
        ax = axes[0, 0]
        ax.plot(range(1, n_iter + 1), result.rms_correction_per_iteration, 'o-', 
                lw=2, markersize=8, color='C0')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('RMS Shift Correction (pixels)')
        ax.set_title('Corrections convergence')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, n_iter + 1))
        ax.set_yscale('log')
        
        # Plot 2: FWHM evolution (starts from 0)
        ax = axes[0, 1]
        self.plot_cutout_centers(ax)
        
        # Plot 3: Shift corrections scatter
        ax = axes[1, 0]
        max_value = np.max(np.abs(result.shift_corrections))*1.1
        colors = plt.cm.viridis(np.linspace(0, 1, n_iter))
        for i, corrections in enumerate(result.shift_corrections):
            ax.scatter(corrections[:, 1], corrections[:, 0], c=[colors[i]], 
                      alpha=0.5, label=f'Iter {i+1}', s=20)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Correction X (pixels)')
        ax.set_ylabel('Correction Y (pixels)')
        ax.set_title('Shift Corrections')
        ax.legend(fontsize=8)
        ax.set_xlim(-max_value,max_value)
        ax.set_ylim(-max_value,max_value)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Correction magnitude histogram
        ax = axes[1, 1]
        for i, corrections in enumerate(result.shift_corrections):
            magnitudes = np.sqrt(corrections[:, 0]**2 + corrections[:, 1]**2)
            ax.hist(magnitudes, bins=20, alpha=0.5, label=f'Iter {i+1}', 
                   edgecolor='black', linewidth=0.5)
        ax.set_xscale("log")
        ax.set_xlabel('Correction Magnitude (pixels)')
        ax.set_ylabel('Count')
        ax.set_title('Correction Distribution')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_cutout_centers(self, ax,
                            iteration: int = -1,
                            show_arrows: bool = True) -> None:
        """Plot center positions of each cutout after shifts."""
        if self._last_refinement is None:
            print("No refinement history. Run refine_shifts() first.")
            return
        
        result = self._last_refinement
        
        if iteration == -1:
            iteration = result.n_iterations - 1
        
        if iteration >= len(result.total_shifts):
            print(f"Invalid iteration {iteration}. Max is {len(result.total_shifts) - 1}")
            return
        
        shifts = result.total_shifts[iteration]
        
        ax.scatter(shifts[:, 1], shifts[:, 0], c='C0', s=30, alpha=0.6, 
                  label=f'Iteration {iteration + 1}')
        
        if show_arrows:
            for i in range(len(shifts)):
                ax.annotate('', xy=(shifts[i, 1], shifts[i, 0]), xytext=(0, 0),
                           arrowprops=dict(arrowstyle='->', color='C0', alpha=0.3, lw=0.5))
        
        ax.scatter([0], [0], c='red', s=100, marker='+', linewidths=2, 
                  label='Image center', zorder=10)
        
        circle = plt.Circle((0, 0), 0.5, fill=False, color='gray', 
                            linestyle='--', alpha=0)
        ax.add_patch(circle)
        
        ax.set_xlabel('Shift X (pixels)')
        ax.set_ylabel('Shift Y (pixels)')
        ax.set_title(f'Cutout Centers - Iteration {iteration + 1}')
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        """mean_x, mean_y = np.mean(shifts, axis=0)
        std_x, std_y = np.std(shifts, axis=0)
        ax.text(0.02, 0.98, 
                f'Mean: ({mean_y:.3f}, {mean_x:.3f})\n'
                f'Std: ({std_y:.3f}, {std_x:.3f})',
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))"""
        
    
    def plot_epsf_evolution(self, 
                            asinh_scale: float = 1e-4,
                            figsize: Tuple[int, int] = None) -> None:
        """Plot ePSF at each iteration side by side."""
        if self._last_refinement is None:
            print("No refinement history. Run refine_shifts() first.")
            return
        
        result = self._last_refinement
        n_epsf = len(result.epsf_history)
        
        if figsize is None:
            figsize = (4 * n_epsf, 4)
        
        fig, axes = plt.subplots(1, n_epsf, figsize=figsize)
        if n_epsf == 1:
            axes = [axes]
        
        for i, (epsf, ax) in enumerate(zip(result.epsf_history, axes)):
            im = ax.imshow(epsf, origin='lower', cmap='magma',
                          norm=AsinhNorm(asinh_scale))
            ax.set_title(f'Iter {i}\nFWHM={result.epsf_fwhm_per_iteration[i]:.2f}')
            ax.set_xlabel('X')
            if i == 0:
                ax.set_ylabel('Y')
        
        plt.tight_layout()
        plt.show()
    
    def plot_epsf(self,
                  log_scale: bool = True,
                  asinh_scale: float = 1e-4,
                  vmin: Optional[float] = None,
                  vmax: Optional[float] = None,
                  figsize: Tuple[int, int] = (8, 7)) -> None:
        """Display the current ePSF."""
        if self.current_epsf is None:
            print("No ePSF available. Run stack() or refine_shifts() first.")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if log_scale:
            norm = AsinhNorm(asinh_scale, vmin=vmin, vmax=vmax)
        else:
            norm = None
        
        im = ax.imshow(self.current_epsf, origin='lower', cmap='magma', 
                       norm=norm, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label='Normalized flux')
        
        fwhm = self.compute_fwhm(self.current_epsf)
        ee = self.compute_encircled_energy(self.current_epsf)
        
        ax.set_title(f'ePSF ({self.get_n_valid()} stars)\n'
                    f'FWHM={fwhm:.2f} px, EE50={ee[0.5]:.2f} px')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        plt.tight_layout()
        plt.show()