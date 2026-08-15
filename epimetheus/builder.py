from dataclasses import dataclass
from typing import Optional, List, Literal, Tuple, Dict
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import map_coordinates
from photutils.centroids import centroid_2dg
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
from drizzle.resample import Drizzle
from astropy.nddata import block_reduce
from astropy.stats import sigma_clipped_stats
import warnings
import os
from astropy.io import fits
import warnings

from .core import (
    StarCutouts,
    process_cutout,
    normalize_cutout,
    build_pixmap,
    center_index,
    harmonic_mean_weight
)

def extract_centered_symmetric(arr: np.ndarray,
                                    center_rc: Tuple[float, float],
                                    shape: Tuple[int, int]) -> np.ndarray:
    """
    Symmetric crop around center_rc (float), with zero padding.
    Rule:
    - pixel-centered center (integer coords)  -> odd crop
    - inter-pixel center (half-integer coords) -> even crop
    """
    cy, cx = map(float, center_rc)
    sy, sx = map(int, shape)

    fy, fx = cy - np.floor(cy), cx - np.floor(cx)
    if np.isclose(fy, 0.0) and np.isclose(fx, 0.0):
        want_odd = True
    elif np.isclose(fy, 0.5) and np.isclose(fx, 0.5):
        want_odd = False
    else:
        raise ValueError(
            f"center_rc={center_rc} must be either integer-like or half-integer-like in both axes."
        )

    if ((sy % 2 == 1) != want_odd) or ((sx % 2 == 1) != want_odd):
        parity = "odd" if want_odd else "even"
        raise ValueError(
            f"center_rc={center_rc} requires a {parity} cutout shape, but shape is {shape}"
        )

    out = np.zeros((sy, sx), dtype=arr.dtype)

    y0 = int(np.floor(cy - (sy - 1) / 2.0))
    x0 = int(np.floor(cx - (sx - 1) / 2.0))
    y1, x1 = y0 + sy, x0 + sx

    ay0, ax0 = max(0, y0), max(0, x0)
    ay1, ax1 = min(arr.shape[0], y1), min(arr.shape[1], x1)

    if ay1 > ay0 and ax1 > ax0:
        oy0, ox0 = ay0 - y0, ax0 - x0
        out[oy0:oy0 + (ay1 - ay0), ox0:ox0 + (ax1 - ax0)] = arr[ay0:ay1, ax0:ax1]

    return out


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

    Resampling backend
    -------------------
    Each star's native-resolution cutout is resampled directly onto the
    final ePSF grid with the `drizzle` package (`drizzle_resample_star`),
    which applies the native->final rescaling, the fitted shift, and the
    rotation (best_pa [+ measured fine rotation]) in a single pass. This
    replaced the older "upsample onto a supersampled grid with
    scipy.ndimage.affine_transform, then block-reduce" approach, which had
    two problems this class now avoids:

      * ``star.scale_factor`` < 1 (final grid finer than the native image)
        was silently mishandled — the "native -> final" ratio is now
        always computed as ``1.0 / star.scale_factor``, which is correct
        for scale_factor either >= 1 or < 1 (see ``drizzle_resample_star``
        in core.py).
      * The intermediate supersampled grid could be off-center by a
        fraction of a pixel whenever the upsampling factor wasn't an
        exact integer. Drizzling straight onto the final grid removes the
        intermediate step entirely, so there's nothing to mis-center.
    """

    DEFAULT_MAX_SHIFT = 5.0  # px (native)
    DEFAULT_FIT_ORDER = 5
    DEFAULT_K_SIGMA = 3.0
    DEFAULT_MAXITERS = 10
    DEFAULT_NORM_RADIUS = 1.0  # arcsec
    DEFAULT_PIXFRAC = 0.6
    DEFAULT_KERNEL = "gaussian"

    def __init__(self, cutouts: StarCutouts, norm_radius: float = DEFAULT_NORM_RADIUS,
                 pixfrac: float = DEFAULT_PIXFRAC, kernel: str = DEFAULT_KERNEL):
        self.cutouts = cutouts
        self.norm_radius = norm_radius
        self.pixfrac = pixfrac
        self.kernel = kernel
        warnings.filterwarnings("ignore", message="Kernel 'gaussian' is not a flux-conserving kernel")
        self.current_epsf: Optional[np.ndarray] = None
        self._last_refinement: Optional[RefinementResult] = None

        # Internal storage (does NOT modify original cutouts)
        self._shifts: Dict[int, Tuple[float, float]] = {}
        self._fit_amplitudes: Dict[int, float] = {}

        #Contains the builded epsf
        self.epsf_angle0 = None

        # Initialize with original values
        self._initialize_from_cutouts()

    def _initialize_from_cutouts(self) -> None:
        """Copy initial shifts and elaborated cutouts from original cutouts."""
        self._shifts.clear()
        self._fit_amplitudes.clear()

        self._pixmaps = {}
        self._native_weight_working = {}
        self._native_flux_scale = {}


        for i, star in enumerate(self.cutouts.stars):
            if star is not None and star.is_valid == 1 and not star.mask:
                # Copy shift (NATIVE pixel units)
                if star.shift is not None:
                    self._shifts[i] = star.shift

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
            and star.cutout is not None
        ]

    def get_n_valid(self) -> int:
        """Return number of valid stars."""
        return len(self.get_valid_indices())

    def _get_shift(self, star_index: int) -> Tuple[float, float]:
        """Get shift (NATIVE pixel units) from internal storage or fall back to original."""
        if star_index in self._shifts:
            return self._shifts[star_index]
        star = self.cutouts.stars[star_index]
        if star is not None and star.shift is not None:
            return star.shift
        return (0.0, 0.0)

    def _align_star_internal(self,
                          star_index: int,
                          shift: Tuple[float, float],
                          final_angle: float = 0.0,
                          fine_rotation: bool = False,
                          normalize: bool = True,
                          norm_radius: float = 1.0) -> None:
        """
        Prepare ONE star for shared drizzle stacking.

        Does NOT resample the star onto the final grid as a standalone
        "final" product. Instead it:
        - computes and caches this star's NATIVE -> FINAL `pixmap`
            (`self._pixmaps[star_index]`), needed both to add this star's
            NATIVE data into the shared multi-image Drizzle object in
            `stack()`, and to translate output-grid pixel rejections back
            into native-pixel weight zeroing;
        - seeds a mutable working copy of the NATIVE weight map
            (`self._native_weight_working[star_index]`), from
            `star.cutout_wht` with `star.cutout_mask` applied. `stack()`'s
            rejection loop progressively zeroes entries in this array; it
            is the ONLY thing rejection is allowed to mutate;
        - drizzles the NATIVE data (single star, single image) onto the
            final grid purely as a per-star STATISTICS image for outlier
            detection in `stack()` -- this is never treated as a final
            combined product on its own.

        ``shift`` is in NATIVE pixel units. `star.cutout` / `star.cutout_wht`
        (the originals) are read but never modified.
        """
        star = self.cutouts.stars[star_index]
        best_pa = star.best_pa if star.best_pa is not None else 0

        conv_factor = 1.0 / float(star.scale_factor)
        rot_angle = best_pa - final_angle
        if fine_rotation and star.rotation is not None:
            rot_angle -= star.rotation

        shift_native = np.asarray(shift, dtype=float)
        shift_out = tuple(shift_native * conv_factor)

        final_size = tuple(int(s)*self.cutouts.upsample_factor for s in self.cutouts.cutout_shape)

        pixmap = build_pixmap(
            shape_in=star.cutout.shape,
            shape_out=final_size,
            shift=shift_out,
            angle_deg=rot_angle,
            scale_factor=conv_factor,
        )

        if star.cutout_wht is not None:
            working_weight = np.asarray(star.cutout_wht, dtype=np.float32).copy()
            working_weight[~np.isfinite(working_weight)] = 0.0
            working_weight[working_weight < 0] = 0.0
        else:
            working_weight = np.ones_like(star.cutout, dtype=np.float32)

        if star.cutout_mask is not None:
            working_weight[star.cutout_mask] = 0.0

        if not hasattr(self, '_pixmaps'):
            self._pixmaps = {}
        if not hasattr(self, '_native_weight_working'):
            self._native_weight_working = {}
        if not hasattr(self, '_native_flux_scale'):
            self._native_flux_scale = {}

        self._pixmaps[star_index] = pixmap
        self._native_weight_working[star_index] = working_weight

        # Single-star statistics drizzle (native -> final), for REJECTION
        # PURPOSES ONLY.
        driz = Drizzle(kernel=self.kernel, out_shape=final_size, fillval="0.0")
        driz.add_image(
            data=np.asarray(star.cutout, dtype=np.float32),
            exptime=1.0,
            pixmap=pixmap,
            weight_map=working_weight,
            pixfrac=self.pixfrac,
            in_units="cps",
        )
        reduced_image = np.nan_to_num(driz.out_img, nan=0.0).astype(np.float64)/conv_factor**2 # scale down area
        reduced_wht = np.nan_to_num(driz.out_wht, nan=0.0).astype(np.float64)*conv_factor**4 # scale down area


        flux_scale = 1.0

        if normalize and star_index in self._fit_amplitudes:
            amp = max(self._fit_amplitudes[star_index], 1e-12)
            flux_scale = 1.0 / amp

            reduced_image = reduced_image * flux_scale
            reduced_wht = reduced_wht / (flux_scale ** 2)

        elif normalize:
            valid = reduced_wht > 0
            flux_init = reduced_image[valid].sum() if np.any(valid) else 0.0

            reduced_image, reduced_wht = normalize_cutout(
                cutout=reduced_image,
                cutout_wht=reduced_wht,
                norm_radius=norm_radius,
                final_resolution=self.cutouts.final_resolution
            )

            flux_after = reduced_image[valid].sum() if np.any(valid) else 1.0

            flux_scale = flux_after/flux_init

        self._native_flux_scale[star_index] = flux_scale
        star.cutout_elaborated_builder = reduced_image
        star.cutout_wht_elaborated_builder = reduced_wht


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
            spline = UnivariateSpline(r_fine, radial_profile, s=0, k=3)

            # Find where profile crosses half_max
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
                return amp * np.exp(-r ** 2 / (2 * sigma ** 2)) + offset

            try:
                p0 = [np.max(radial_profile) - np.min(radial_profile),
                      2.0,
                      np.min(radial_profile)]
                popt, _ = curve_fit(gaussian, r_fine, radial_profile, p0=p0, maxfev=2000)
                sigma = abs(popt[1])
                return 2.355 * sigma  # FWHM = 2*sqrt(2*ln(2)) * sigma
            except Exception:
                # Fallback to half_max method
                return self.compute_fwhm(epsf, method='half_max')

    def compute_encircled_energy(self,
                                  epsf: np.ndarray,
                                  fractions: List[float] = [0.5, 0.8, 0.9]) -> dict:
        """Compute encircled energy radii."""
        center = (np.array(epsf.shape) - 1) / 2.0
        y, x = np.ogrid[:epsf.shape[0], :epsf.shape[1]]
        r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)

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
            results[frac] = r_sorted[min(idx, len(r_sorted) - 1)]

        return results

    # =========================================================================
    # STACKING
    # =========================================================================

    def stack(self,
            rejection_mode: Literal['none', 'weighted', 'mad'] = 'mad',
            k_sigma: float = DEFAULT_K_SIGMA,
            maxiters_rejection: int = 2,
            remove_background: bool = True,
            verbose: bool = True) -> np.ndarray:
        """
        Combine all stars prepared by `_align_star_internal` into a single
        ePSF.

        Rejection (per output pixel, across stars) is driven by each star's
        single-image statistics drizzle (`star.cutout_elaborated_builder`). An
        output-pixel outlier for a given star is converted back to NATIVE
        pixels via that star's forward `pixmap` and zeroed in
        `self._native_weight_working[star_idx]` -- `star.cutout` /
        `star.cutout_wht` are never touched.

        The FINAL image is produced by ONE shared `Drizzle` object: every
        star's ORIGINAL NATIVE cutout is added via its own `add_image` call
        (native data, rejection-pruned native weight, native->final pixmap),
        all accumulating onto the same output grid in a single pass. This is
        what actually uses the cross-star sub-pixel dithering -- collapsing
        each star to the final grid separately first and only then
        co-adding would not.
        """
        valid_indices = [
            i for i in self.get_valid_indices()
            if getattr(self.cutouts.stars[i], 'cutout_elaborated_builder', None) is not None
            and i in getattr(self, '_pixmaps', {})
        ]

        if len(valid_indices) == 0:
            raise ValueError("No aligned stars available for stacking "
                            "(call _align_star_internal first)")

        n_stars = len(valid_indices)
        final_size = tuple(int(s)*self.cutouts.upsample_factor for s in self.cutouts.cutout_shape)

        # Per-star statistics images/weights on the output grid -- used ONLY
        # to decide which NATIVE pixels to reject, never combined directly.
        stat_images = np.nan_to_num(
            np.stack([self.cutouts.stars[i].cutout_elaborated_builder for i in valid_indices], axis=0),
            nan=0.0
        ).astype(np.float64)
        stat_weights = np.stack(
            [np.nan_to_num(self.cutouts.stars[i].cutout_wht_elaborated_builder, nan=0.0) for i in valid_indices],
            axis=0
        ).astype(np.float64)
        stat_weights[stat_weights < 0] = 0.0

        for iteration in range(max(1, maxiters_rejection)):
            if rejection_mode == 'none' or n_stars < 3:
                break

            active = stat_weights > 0
            n_active_before = int(np.sum(active))
            if n_active_before == 0:
                break

            if rejection_mode == 'mad':
                masked = np.where(active, stat_images, np.nan)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        'ignore', message='All-NaN slice encountered', category=RuntimeWarning
                    )
                    with np.errstate(invalid='ignore'):
                        center = np.nanmedian(masked, axis=0)
                        sigma = 1.4826 * np.nanmedian(np.abs(masked - center[None, ...]), axis=0)

            else:  # 'weighted'
                with np.errstate(invalid='ignore', divide='ignore'):
                    wsum = np.sum(stat_weights, axis=0)
                    center = np.divide(np.sum(stat_weights * stat_images, axis=0), wsum,
                                        out=np.zeros(final_size), where=wsum > 0)
                    resid = np.where(active, stat_images - center[None, ...], np.nan)
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=RuntimeWarning)
                        sigma = np.nanstd(resid, axis=0)

            n_contrib = np.sum(active, axis=0)
            sigma = np.where((sigma > 0) & np.isfinite(sigma) & (n_contrib >= 2), sigma, np.inf)

            deviation = np.abs(stat_images - center[None, ...])
            outliers = active & (deviation > k_sigma * sigma[None, ...])

            if not np.any(outliers):
                if verbose:
                    print(f"  Rejection converged after {iteration + 1} iteration(s)")
                break

            for pos, star_idx in enumerate(valid_indices):
                star_outliers = outliers[pos]
                if not np.any(star_outliers):
                    continue

                stat_weights[pos][star_outliers] = 0.0
                stat_images[pos][star_outliers] = 0.0

                pixmap = self._pixmaps[star_idx]
                out_x = np.round(pixmap[..., 0]).astype(int)
                out_y = np.round(pixmap[..., 1]).astype(int)
                in_bounds = (
                    (out_x >= 0) & (out_x < final_size[1]) &
                    (out_y >= 0) & (out_y < final_size[0])
                )
                native_reject = np.zeros(out_x.shape, dtype=bool)
                native_reject[in_bounds] = star_outliers[out_y[in_bounds], out_x[in_bounds]]

                self._native_weight_working[star_idx][native_reject] = 0.0

            n_active_after = int(np.sum(stat_weights > 0))
            if verbose:
                print(f"  Iter {iteration + 1}: rejected "
                    f"{n_active_before - n_active_after} pixel-contribution(s)")

            if n_active_after == n_active_before:
                break

        # ------------------------------------------------------------------
        # Final combination: ONE shared Drizzle object, every star's NATIVE
        # data added together in a single accumulation pass.
        # ------------------------------------------------------------------

        max_inv_k2 = max(1.0 / (self._native_flux_scale.get(i, 1.0) * star.scale_factor**2)**2 for i, star in enumerate(self.cutouts.stars) if i  in valid_indices)
        

        driz = Drizzle(kernel=self.kernel, out_shape=final_size, fillval="0.0")
        for star_idx in valid_indices:
            star = self.cutouts.stars[star_idx]
            flux_scale = self._native_flux_scale.get(star_idx, 1.0)


            k_factor = flux_scale * star.scale_factor**2


            # This normalizes the weights so the maximum weight multiplier is around 1.0
            norm_weight_modifier = (1.0 / (k_factor ** 2)) / max_inv_k2

            """driz_2 = Drizzle(kernel=self.kernel, out_shape=final_size, fillval="0.0")
            driz_2.add_image(
                data=np.asarray(star.cutout, dtype=np.float32) * k_factor,
                exptime=1.0,
                pixmap=self._pixmaps[star_idx],
                weight_map=np.asarray(self._native_weight_working[star_idx], dtype=np.float32) * norm_weight_modifier,
                pixfrac=self.pixfrac,
                in_units="cps",
            )"""

            driz.add_image(
                data=np.asarray(star.cutout, dtype=np.float32) * k_factor,
                exptime=1.0,
                pixmap=self._pixmaps[star_idx],
                weight_map = np.asarray(self._native_weight_working[star_idx], dtype=np.float32) * norm_weight_modifier,
                pixfrac=self.pixfrac,
                in_units="cps",
            )

            """plt.figure()         
            plt.subplot(131)
            plt.imshow(driz_2.out_img,norm=AsinhNorm(1e-7,0,3e-3),origin="lower",cmap="magma")
            plt.subplot(132)
            plt.imshow(block_reduce(driz_2.out_img,4)[90:110,90:110],origin="lower",cmap="magma")
            plt.subplot(133)
            plt.imshow(block_reduce(driz.out_img,4)[90:110,90:110],origin="lower",cmap="magma")
            plt.show()

            print(np.median(driz_2.out_wht), np.median(star.cutout_wht), star.scale_factor)"""

            
        stacked = np.nan_to_num(driz.out_img, nan=0.0).astype(np.float64)

        """plt.figure()      
        plt.title("stacked")   
        plt.imshow(stacked,norm=AsinhNorm(1e-7,0,3e-3),origin="lower",cmap="magma")
        plt.colorbar()
        plt.show()"""
        
        if remove_background:
            __, bkg, __ = sigma_clipped_stats(stacked)
            stacked = stacked - bkg

        stacked = stacked/np.sum(stacked)

        # Persist rejection-pruned stat weights for inspection.
        for pos, star_idx in enumerate(valid_indices):
            self.cutouts.stars[star_idx].cutout_wht_elaborated_builder = stat_weights[pos]
            self.cutouts.stars[star_idx].cutout_weights_rej_builder = np.asarray(self._native_weight_working[star_idx], dtype=np.float32)

        return stacked

    # =========================================================================
    # SHIFT FITTING
    # =========================================================================
    def _fit_shift_to_original(self,
                            star_index: int,
                            epsf: np.ndarray,
                            fwhm: float,
                            fwhm_fraction: float = 3.0,
                            max_shift: float = DEFAULT_MAX_SHIFT,
                            fit_order: int = DEFAULT_FIT_ORDER) -> Tuple[float, float]:
        """
        Fit the TOTAL shift (in NATIVE pixels) needed to align the original
        native-resolution cutout to the current ePSF.

        1. crop a high-resolution patch from the ePSF,
        2. rotate it into the star's native orientation, still at high-res,
        3. apply the trial subpixel shift on that high-res grid,
        4. block-reduce back to native resolution,
        5. fit amplitude analytically (background offset fixed to 0).

        Notes
        -----
        - Returned shifts are in NATIVE pixel units.
        - The fitted amplitude is stored in `self._fit_amplitudes[star_index]`.
        If you want to use it for normalization, scale the star by 1/amplitude.
        - Background offset is NOT fitted; it is fixed to 0.
        - This version is consistent with refinement done in the canonical
        `final_angle=0` frame with `fine_rotation=False`, i.e. the inverse
        rotation applied here is just `+best_pa`.
        """

        star = self.cutouts.stars[star_index]

        if star is None or star.cutout is None:
            raise ValueError("Star is not extracted, cannot fit.")
        
        centroid_radius = max(int(round((fwhm * fwhm_fraction * star.scale_factor - 1)/2)),1)
        #so if images of different scales the physical psf size in the same , with different number of pixels. Indeed fwhm is measured on the upsample images

        # Native -> ePSF magnification
        native_to_epsf = 1.0 / float(star.scale_factor)

        # Use integer block reduction only when the upsampling is effectively integer
        upsamp_int = max(1,int(round(native_to_epsf)))
        use_block_reduce = (
            upsamp_int > 1 and
            np.isclose(native_to_epsf, upsamp_int, rtol=0.0, atol=1e-6)
        )

        # ------------------------------------------------------------------
        # Native fitting patch from the original cutout
        # ------------------------------------------------------------------

        if star.scale_factor>1:
            if np.isclose(star.scale_factor, int(np.round(star.scale_factor))):
                data_reducing_factor = int(np.round(star.scale_factor))
                img = block_reduce(star.cutout, data_reducing_factor , func = np.sum)

                if star.cutout_wht is not None:
                    img_wht = block_reduce(star.cutout_wht, data_reducing_factor, func = harmonic_mean_weight)
                else:
                    img_wht = None

                if star.cutout_mask is not None:
                    img_mask = block_reduce(star.cutout_mask, data_reducing_factor, func = np.min)
                else:
                    img_mask = None
            else:
                raise NotImplementedError(f"star.scale_factor is {star.scale_factor} which is not integer")
        else:
            data_reducing_factor = 1
            img = star.cutout
            img_wht = star.cutout_wht
            img_mask = star.cutout_mask


        center_native = center_index(img.shape[0]), center_index(img.shape[1])

        if img.shape[0] % 2 == 0:
            patch_size = 2 * centroid_radius + 2
        else:
            patch_size = 2 * centroid_radius + 1


        if np.shape(img)[0] != np.shape(img)[1]:
            self._fit_amplitudes[star_index] = 1e-12
            return 0.0, 0.0, (np.zeros((patch_size,patch_size)), np.zeros((patch_size,patch_size)))


        data = extract_centered_symmetric(img, center_rc=center_native, shape=(patch_size,patch_size)).astype(np.float64)


        if img_mask is not None:
            mask_data = extract_centered_symmetric(img_mask, center_rc=center_native, shape=(patch_size,patch_size)).astype(bool)
        else:
            mask_data = np.zeros_like(data, dtype=bool)

        if img_wht is not None:
            wht = extract_centered_symmetric(img_wht, center_rc=center_native, shape=(patch_size,patch_size)).astype(np.float64)
            wht[~np.isfinite(wht)] = 0.0
            wht[wht < 0] = 0.0
        else:
            wht = np.ones_like(data, dtype=np.float64)

        wht[mask_data] = 0.0
        valid = wht > 0

        """plt.figure()
        plt.subplot(131)
        plt.imshow(data)
        plt.subplot(132)
        plt.imshow(mask_data)
        plt.subplot(133)
        plt.imshow(wht)
        plt.show()"""

        if not np.any(valid):
            if not hasattr(self, "_fit_amplitudes"):
                self._fit_amplitudes = {}
            self._fit_amplitudes[star_index] = 1.0
            return 0.0, 0.0, (data, np.zeros_like(data))

        wht_sum = np.sum(wht)
        wht_norm = wht / wht_sum if wht_sum > 0 else np.ones_like(wht) / wht.size

        # Consistent with refine_shifts(... final_angle=0, fine_rotation=False)
        rot_angle = star.best_pa if star.best_pa is not None else 0.0

        # ------------------------------------------------------------------
        # Build a high-resolution, native-orientation PSF patch
        # ------------------------------------------------------------------
        margin_interp = int(np.ceil(fit_order / 2.0)) + 2
        hr_core_size = max(patch_size, 3) * upsamp_int

        epsf_radius = int(np.ceil((fwhm * fwhm_fraction - 1) / 2 + max_shift * self.cutouts.upsample_factor)) + margin_interp

        epsf_center = center_index(epsf.shape[0]), center_index(epsf.shape[1])
        center_is_integer = np.isclose(epsf_center[0] % 1.0, 0.0)

        epsf_cutout_size = max(
            int(np.ceil(epsf_radius / self.cutouts.upsample_factor)) * self.cutouts.upsample_factor,
            hr_core_size
        )

        if ((epsf_cutout_size % 2 == 1) != center_is_integer):
            epsf_cutout_size += 1

        epsf_crop_hr = extract_centered_symmetric(
            epsf,
            center_rc=(epsf_center[0], epsf_center[1]),
            shape=(epsf_cutout_size,epsf_cutout_size)
        )

        if use_block_reduce:

            # Rotate to the star's native orientation, but KEEP the high-res grid
            psf_hr_native = process_cutout(
                data=epsf_crop_hr,
                shift=(0.0, 0.0),
                angle=rot_angle,
                scale_factor=1.0,
                output_shape=epsf_crop_hr.shape,
                order=fit_order
            )

            center_hr = center_index(epsf_cutout_size)
            pad0 = int(np.floor(center_hr - (hr_core_size - 1) / 2.0))
            core_slice = (slice(pad0, pad0 + hr_core_size), slice(pad0, pad0 + hr_core_size))

            def build_model(shift_x_native: float, shift_y_native: float) -> np.ndarray:
                # Apply shift on the HIGH-RES grid first
                shift_x_hr = shift_x_native * upsamp_int
                shift_y_hr = shift_y_native * upsamp_int

                shifted_hr = process_cutout(
                    data=psf_hr_native,
                    shift=(shift_y_hr, shift_x_hr),
                    angle=0.0,
                    scale_factor=1.0,
                    output_shape=psf_hr_native.shape,
                    order=fit_order
                )

                core_hr = shifted_hr[core_slice]

                model_native = block_reduce(
                    core_hr,
                    block_size=(upsamp_int, upsamp_int),
                    func=np.sum
                )
                model_native = np.asarray(model_native, dtype=np.float64)

                if model_native.shape != (patch_size, patch_size):
                    raise ValueError(
                        f"[star {star_index}] block_reduce produced shape "
                        f"{model_native.shape}, expected ({patch_size}, {patch_size}). "
                        f"hr_core_size={hr_core_size}, upsamp_int={upsamp_int}."
                    )

                return model_native

        else:
            scale_factor = min(1, star.scale_factor)#scale_factor>1 are already computed separately by blocking reduce the image

            psf_hr_native = process_cutout(
                data=epsf_crop_hr,
                shift=(0.0, 0.0),
                angle=rot_angle,
                scale_factor=scale_factor,  # fine -> native
                output_shape=(patch_size, patch_size),
                order=fit_order
            )

            def build_model(shift_x_native: float, shift_y_native: float) -> np.ndarray:
                model_native = process_cutout(
                    data=psf_hr_native,
                    shift=(shift_y_native, shift_x_native),
                    angle=0.0,
                    scale_factor=1.0,
                    output_shape=psf_hr_native.shape,
                    order=fit_order
                )
                return np.asarray(model_native, dtype=np.float64)

        """plt.figure()
        plt.subplot(131)
        plt.suptitle("PSF")
        plt.imshow(epsf_crop_hr)
        plt.subplot(132)
        plt.imshow(psf_hr_native)
        plt.subplot(133)
        plt.imshow(data)
        plt.show()"""

        # ------------------------------------------------------------------
        # Weighted linear solve for amplitude at fixed shift
        # (background offset is fixed to 0, NOT fitted)
        # ------------------------------------------------------------------
        def solve_amplitude(model: np.ndarray) -> float:
            m = model[valid].ravel()
            d = data[valid].ravel()
            ww = wht[valid].ravel()

            if m.size == 0:
                return 1.0

            # Weighted least squares for a single scale parameter:
            # minimize sum(w * (d - amp*m)^2)  ->  amp = sum(w*m*d) / sum(w*m^2)
            denom = np.sum(ww * m * m)
            if denom > 0:
                amp = float(np.sum(ww * m * d) / denom)
            else:
                amp = 1.0

            # Optional positivity constraint on amplitude
            if not np.isfinite(amp) or amp < 0:
                amp = 0.0

            return amp

        def residual(shift_params: np.ndarray) -> float:
            shift_x, shift_y = shift_params
            model = build_model(shift_x, shift_y)
            amp = solve_amplitude(model)
            resid = data - amp * model
            return float(np.sum(wht_norm * resid ** 2))

        # ------------------------------------------------------------------
        # Optimize shift only; amplitude solved analytically, offset = 0
        # ------------------------------------------------------------------
        max_shift_star = max_shift * self.cutouts.upsample_factor / upsamp_int
        #same shift in science for each star

        result = minimize(
            residual,
            x0=np.array([0.0, 0.0], dtype=float),
            method='Powell',
            bounds=[(-max_shift_star, max_shift_star), (-max_shift_star, max_shift_star)]
        )

        if result.success:
            shift_x, shift_y = float(result.x[0]), float(result.x[1])
            best_model = build_model(shift_x, shift_y)
            best_amp = solve_amplitude(best_model)

            """print(star_index, shift_x,shift_y)
            fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(8,2))            
            im1 = ax1.imshow(data/np.sum(data),origin="lower",cmap="magma")
            ax2.imshow((best_model*best_amp)/np.sum(data),origin="lower",cmap="magma")
            try:
                ax3.imshow(block_reduce(psf_hr_native[core_slice],upsamp_int, np.sum)/np.sum(psf_hr_native[core_slice]),origin="lower",cmap="magma")
                shift_img = process_cutout(data,(-shift_y, -shift_x))
                ax4.imshow(shift_img/np.sum(data),origin="lower",cmap="magma")
            except:
                pass
            plt.show()"""

        else:
            # Fallback: centroid on native cutout patch
            if star.cutout_wht is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    error = np.sqrt(1.0 / (wht + 1e-50))
                    finite = np.isfinite(error)
                    if np.any(finite):
                        error[~finite] = np.nanmax(error[finite])
                    else:
                        error = None
            else:
                error = None

            cx, cy = centroid_2dg(data, error=error, mask=mask_data)
            shift_x = float(cx - center_index(patch_size))
            shift_y = float(cy - center_index(patch_size))

            best_model = build_model(shift_x, shift_y)
            best_amp = solve_amplitude(best_model)

            """fig, (ax1,ax2) = plt.subplots(1,2)            
            im1 = ax1.imshow(data/np.sum(data),norm=AsinhNorm(1e-5,0,0.1),origin="lower",cmap="magma")
            ax2.imshow((best_model*best_amp)/np.sum(data),norm=AsinhNorm(1e-5,0,0.1),origin="lower",cmap="magma")
            plt.show()"""

        #residuals = data-(best_model*best_amp)

        # Store amplitude for later normalization use
        if not hasattr(self, "_fit_amplitudes"):
            self._fit_amplitudes = {}
        self._fit_amplitudes[star_index] = max(float(best_amp), 1e-12)

        # fitted model shift -> total star alignment shift
        # return to the data original shift frame
        return -shift_y*data_reducing_factor, -shift_x*data_reducing_factor, (data, best_model*best_amp)

    def _recenter_epsf(self,
                    epsf: np.ndarray,
                    fwhm: float,
                    fwhm_fraction: float = 1.0,
                    order: int = 3,
                    max_iter: int = 10,
                    shift_tol: float = 1e-4) -> Tuple[np.ndarray, Tuple[float, float]]:
        """
        Recenter ePSF so its centroid (from a 2D Gaussian fit) sits at the
        array's geometric center, iterating until convergence.

        Target parity is fixed by array shape:
        - odd size  -> geometric center is an integer pixel (pixel-centered)
        - even size -> geometric center is a half-integer (inter-pixel corner)

        The fitting window is built with matching parity so it is always
        symmetric around that fixed target, regardless of where the star
        currently sits.

        At each iteration the centroid offset is measured on the current
        recentered estimate; the *cumulative* offset is then re-applied to the
        original (unshifted) array, so the star only ever passes through a
        single resampling relative to its original position (avoiding repeated
        interpolation blur from re-shifting an already-shifted image).

        Iterates until the incremental offset is below `shift_tol` (in pixels)
        or `max_iter` iterations are reached.
        """
        epsf = np.asarray(epsf, dtype=np.float64)
        epsf_copy = epsf.copy()
        epsf_copy[~np.isfinite(epsf_copy)] = 0.0

        ny, nx = epsf_copy.shape
        geom_center = (np.array(epsf_copy.shape, dtype=float) - 1.0) / 2.0
        cy, cx = geom_center

        # Parity is a fixed property of the array shape, not of where the star sits.
        want_odd_y = (ny % 2 == 1)
        want_odd_x = (nx % 2 == 1)
        if want_odd_y != want_odd_x:
            raise ValueError(
                f"epsf shape {epsf_copy.shape} has mixed odd/even axes; "
                "extract_centered_symmetric requires matching parity on both axes."
            )
        want_odd = want_odd_y

        total = np.maximum(epsf_copy, 0.0).sum()
        if total <= 0:
            return epsf_copy.copy(), (0.0, 0.0)

        # Window size, forced to match the array's fixed parity
        half_width = max(fwhm_fraction * fwhm, 2.0)
        box_size = int(np.ceil(2 * half_width)) + 1

        if (want_odd and box_size % 2 == 0) or ((not want_odd) and box_size % 2 == 1):
            box_size += 1

        # Crop origin in the full-array frame (matches extract_centered_symmetric's
        # own placement formula) so we can convert local fits back to absolute coords.
        y0 = int(np.floor(cy - (box_size - 1) / 2.0))
        x0 = int(np.floor(cx - (box_size - 1) / 2.0))

        def _measure_offset(image: np.ndarray) -> Tuple[float, float]:
            """Return (offset_y, offset_x) of image's centroid from geom_center,
            or None if the measurement failed."""
            core = extract_centered_symmetric(image, (cy, cx), (box_size, box_size))
            if core.sum() <= 0:
                return None

            try:
                x_local, y_local = centroid_2dg(core)
            except Exception:
                return None

            if not (np.isfinite(x_local) and np.isfinite(y_local)):
                return None

            centroid_y = y_local + y0
            centroid_x = x_local + x0
            return (centroid_y - cy, centroid_x - cx)

        current = epsf_copy
        total_offset_y, total_offset_x = 0.0, 0.0
        converged = False

        for _ in range(max_iter):
            measured = _measure_offset(current)
            if measured is None:
                # Fit failed on the first iteration -> bail out safely as before.
                if total_offset_y == 0.0 and total_offset_x == 0.0:
                    return epsf.copy(), (0.0, 0.0)
                # Otherwise keep the best estimate obtained so far.
                break

            offset_y, offset_x = measured
            step = np.hypot(offset_y, offset_x)

            if step < shift_tol:
                converged = True
                break

            total_offset_y += offset_y
            total_offset_x += offset_x

            # Re-derive the recentered image from the ORIGINAL array using the
            # cumulative offset, so only one resampling is ever applied.
            current = process_cutout(
                data=epsf_copy,
                shift=(-total_offset_y, -total_offset_x),
                angle=0.0,
                scale_factor=1.0,
                output_shape=epsf_copy.shape,
                order=order
            )

        if total_offset_y == 0.0 and total_offset_x == 0.0:
            # Either converged immediately, or nothing ever moved.
            return epsf_copy.copy(), (0.0, 0.0)

        return current, (total_offset_y, total_offset_x)
    
    def _propagate_epsf_recentering_to_shifts(self,
                                            offset_epsf: Tuple[float, float],
                                            star_indices: Optional[List[int]] = None) -> None:
        """
        Propagate an ePSF-frame recentering into the stored native-pixel shifts.

        Parameters
        ----------
        offset_epsf : Tuple[float, float]
            (offset_y, offset_x) returned by `_recenter_epsf`, i.e.
            centroid - geometric_center, in ePSF/final-grid pixels.
        star_indices : list[int] or None
            Which stars to update. If None, update all valid stars.
        """
        if star_indices is None:
            star_indices = self.get_valid_indices()

        offset_epsf = np.asarray(offset_epsf, dtype=float)

        for star_idx in star_indices:
            star = self.cutouts.stars[star_idx]
            old_shift = np.asarray(self._get_shift(star_idx), dtype=float)

            # Convert ePSF-pixel offset -> native-pixel offset
            offset_native = offset_epsf * float(star.scale_factor)

            # Important: add the raw centroid offset here.
            # This keeps the stored shifts consistent with the recentered PSF frame.
            new_shift = old_shift - offset_native

            self._shifts[star_idx] = (float(new_shift[0]), float(new_shift[1]))

    def refine_shifts(self,
                       n_iterations: int = DEFAULT_MAXITERS,
                       rejection_mode: Literal['none', 'weighted', 'mad'] = 'mad',
                       k_sigma: float = DEFAULT_K_SIGMA,
                       max_shift: float = DEFAULT_MAX_SHIFT,
                       fit_order: int = DEFAULT_FIT_ORDER,
                       maxiters_rejection: int = 2,
                       convergence_threshold: float = 0.01,
                       fwhm_fraction: int = 3, #the cutout size in  fwhm to align fit the psf to te stars
                       recenter_epsf: bool = True,
                       remove_background: bool = True,
                       normalize: bool = True,
                       verbose: bool = True) -> RefinementResult:
        """
        Iteratively refine star shifts by fitting original cutouts to stacked ePSF.

        This does NOT modify the original cutouts. All changes are
        stored internally in self._shifts and self._elaborated.
        """
        self._initialize_from_cutouts()

        valid_indices = self.get_valid_indices()
        n_valid = len(valid_indices)

        if n_valid == 0:
            raise ValueError("No valid stars for refinement")

        if verbose:
            print(f"{'=' * 60}")
            print("Starting iterative shift refinement")
            print(f"  Valid stars: {n_valid}")
            print(f"  Max iterations: {n_iterations}")
            print(f"  Rejection: {rejection_mode}, k_sigma={k_sigma}")
            print(f"  Recenter ePSF: {recenter_epsf}")
            print(f"  Convergence threshold: {convergence_threshold} px")
            print(f"{'=' * 60}")


        for star_idx in valid_indices:
            self._align_star_internal(
                star_idx,
                shift = self._get_shift(star_idx),
                final_angle=0.0,
                fine_rotation=False,
                normalize=normalize,
                norm_radius=self.norm_radius
            )

        epsf_initial = self.stack(rejection_mode=rejection_mode, k_sigma=k_sigma,
                                   maxiters_rejection=maxiters_rejection, verbose=verbose, remove_background=remove_background)
        
        fwhm_initial = self.compute_fwhm(epsf_initial)
        
        if recenter_epsf:
            epsf_initial, initial_offset = self._recenter_epsf(epsf_initial, fwhm=fwhm_initial)
            if verbose:
                print(f"  Initial ePSF recentered by: ({initial_offset[0]:.4f}, {initial_offset[1]:.4f}) px")

        self.current_epsf = epsf_initial


        if verbose:
            print(f"  Initial FWHM: {fwhm_initial:.3f} px")

        shift_corrections = []
        total_shifts = []
        rms_history = []
        fwhm_history = [fwhm_initial]
        epsf_history = [epsf_initial.copy()]
        epsf_offsets = [] # Track offsets for debugging
        converged = False

        for iteration in range(n_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1}/{n_iterations} ---")

            epsf = np.copy(self.current_epsf)

            corrections = np.zeros((n_valid, 2))
            new_shifts = np.zeros((n_valid, 2))

            iterator = tqdm(enumerate(valid_indices), total=n_valid,
                             desc="Fitting shifts", disable=not verbose)

            for i, star_idx in iterator:
                old_shift = np.array(self._get_shift(star_idx))

                fitted_y, fitted_x, __ = self._fit_shift_to_original(
                    star_idx, epsf,
                    max_shift=max_shift,
                    fit_order=fit_order,
                    fwhm=fwhm_history[-1],
                    fwhm_fraction = fwhm_fraction
                )
                new_shift = np.array([fitted_y, fitted_x])

                correction = new_shift - old_shift
                corrections[i] = correction
                new_shifts[i] = new_shift

                self._shifts[star_idx] = (fitted_y, fitted_x)

            shift_corrections.append(corrections.copy())
            total_shifts.append(new_shifts.copy())

            if verbose:
                print("Updating elaborated cutouts...")


            for star_idx in valid_indices:
                self._align_star_internal(
                    star_idx,
                    self._shifts[star_idx],
                    final_angle=0.0,
                    fine_rotation=False,
                    normalize=normalize,
                    norm_radius=self.norm_radius
                )
            epsf = self.stack(
                rejection_mode=rejection_mode,
                k_sigma=k_sigma,
                maxiters_rejection=maxiters_rejection,
                remove_background=remove_background,
                verbose=verbose
            )

            """plt.imshow(epsf,norm=AsinhNorm(1e-5),origin="lower",cmap="magma")
            plt.colorbar()
            print(np.sum(epsf))
            plt.axhline(102.5)
            plt.axvline(102.5)
            plt.show()"""

            fwhm = self.compute_fwhm(epsf)

            if recenter_epsf:
                
                epsf_rec, offset = self._recenter_epsf(epsf, fwhm=fwhm)
                epsf_offsets.append(offset)
                if verbose:
                    print(f"  ePSF recentered by: ({offset[0]:.4f}, {offset[1]:.4f}) px")

                
                """plt.subplot(121)
                plt.imshow(epsf[180:220,180:220])
                plt.subplot(122)
                plt.imshow(epsf_rec[180:220,180:220])
                plt.show()"""

                epsf = epsf_rec

            self.current_epsf = epsf

            epsf_history.append(epsf.copy())

            fwhm_history.append(fwhm)
            if verbose:
                print(f"  ePSF FWHM: {fwhm:.3f} px")

            # Check convergence
            rms_correction = np.sqrt(np.mean(corrections ** 2))
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

        if recenter_epsf and len(epsf_offsets) > 0:
                self._propagate_epsf_recentering_to_shifts(np.array(epsf_offsets[-1]), valid_indices)   # keep shifts consistent with current_epsf
        

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
            print(f"\n{'=' * 60}")
            print(f"Refinement complete: {result.n_iterations} iterations")
            print(f"Converged: {result.converged}")
            print(f"{'=' * 60}")

        return result

    # =========================================================================
    # MULTI-ANGLE ePSF
    # =========================================================================
    def build_multi_angle_epsf(self,
                               angles: Optional[Dict[str, float] | List[float]] = None,
                               output_shape: Optional[Tuple[int, int] | np.ndarray] = None,
                               rejection_mode: Literal['none', 'weighted', 'mad'] = 'mad',
                               k_sigma: float = DEFAULT_K_SIGMA,
                               maxiters_rejection: int = 2,
                               fine_rotation: bool = False,
                               remove_background: bool = True,
                               normalize: bool = True,
                               verbose: bool = True,
                               show_plots: bool = True,
                               save_fits: bool = True) -> Dict[str, np.ndarray]:
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
        maxiters_rejection : int
            Maximum iterations for rejection algorithm. Default is 2.
        fine_rotation : bool
            If True, apply fine rotation correction. Default is False.
        normalize : bool
            If True, normalize each cutout by aperture flux. Default is True.
        verbose : bool
            Print progress. Default is True.
        show_plots : bool
            If True, displays visual plots using matplotlib. Default is True.
        save_fits : bool
            If True, saves FITS files into the cache directory. Default is True.
        base_name : str
            Prefix token for output filenames. Default is "epsf".

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping angle names to ePSF arrays.
        """
        if angles is None:
            angles = [0]

        if isinstance(angles, list):
            angles_dict = {f"angle_{int(a)}": float(a) for a in angles}
        else:
            angles_dict = angles

        if 0 not in angles_dict.values():
            angles_dict["angle_0"] = 0

        valid_indices = self.get_valid_indices()

        if len(valid_indices) == 0:
            raise ValueError("No valid stars for ePSF building")

        crop_slice = None
        if output_shape is not None:
            output_shape = np.asarray(output_shape).copy()
            original_shape = np.asarray(self.cutouts.cutout_shape).copy()

            if np.any(output_shape > original_shape):
                raise ValueError(
                    f"output_shape {tuple(output_shape)} cannot be larger than "
                    f"original shape {tuple(original_shape)}"
                )
            
            # transform to upsample sizes
            original_shape *= self.cutouts.upsample_factor
            output_shape *= self.cutouts.upsample_factor

            start = np.array([
                int(np.floor(center_index(original_shape[0]) - (output_shape[0] - 1) / 2.0)),
                int(np.floor(center_index(original_shape[1]) - (output_shape[1] - 1) / 2.0)),
            ])
            end = start + output_shape
            crop_slice = (slice(start[0], end[0]), slice(start[1], end[1]))


        if verbose:
            print(f"{'=' * 60}")
            print("Building multi-angle ePSF")
            print(f"  Angles: {list(angles_dict.keys())}")
            print(f"  Valid stars: {len(valid_indices)}")
            if output_shape is not None:
                print(f"  Output shape: {tuple(output_shape)}")
            print(f"{'=' * 60}")

        results = {}
        factor = getattr(self.cutouts, 'upsample_factor', 1)

        # Determine target directory
        output_dir = os.path.join(getattr(self.cutouts, 'cache_dir'), "output_psf/")

        if save_fits:
            os.makedirs(output_dir, exist_ok=True)

        for name, target_angle in angles_dict.items():
            if verbose:
                print(f"\n--- {name} (PA = {target_angle}°) ---")

            for star_idx in valid_indices:
                shift = np.array(self._get_shift(star_idx), dtype=float)

                self._align_star_internal(
                    star_idx,
                    tuple(shift),
                    final_angle=target_angle,
                    fine_rotation=fine_rotation,
                    normalize=normalize
                )

            epsf = self.stack(
                rejection_mode=rejection_mode,
                k_sigma=k_sigma,
                maxiters_rejection=maxiters_rejection,
                remove_background=remove_background,
                verbose=verbose
            )

            if crop_slice is not None:
                epsf = epsf[crop_slice]

            s = np.sum(epsf)
            if s > 0:
                epsf = epsf / s

            results[name] = epsf.copy()

            if verbose:
                fwhm = self.compute_fwhm(epsf)
                print(f"  FWHM: {fwhm:.3f} px")
                print(f"  Shape: {epsf.shape}")

            if target_angle==0:
                self.epsf_angle0=epsf

            # --- SAVE FITS SECTION ---
            if save_fits:
                base_name = self.cutouts.name
                if factor > 1:
                    # Save upsampled
                    outpath_upsampled = os.path.join(output_dir, f"{base_name}_{name}_x{factor}sampled.fits")
                    fits.PrimaryHDU(data=epsf).writeto(outpath_upsampled, overwrite=True)
                    
                    # Save downscaled native resolution
                    epsf_red = block_reduce(epsf, factor)
                    outpath_native = os.path.join(output_dir, f"{base_name}_{name}.fits")
                    fits.PrimaryHDU(data=epsf_red).writeto(outpath_native, overwrite=True)
                else:
                    # Save native resolution directly
                    outpath_native = os.path.join(output_dir, f"{base_name}_{name}.fits")
                    fits.PrimaryHDU(data=epsf).writeto(outpath_native, overwrite=True)

        if verbose and save_fits:
            print(f"\nSaved {len(results)} ePSF sets to {output_dir}")

        # --- PLOTTING SECTION ---
        if show_plots and len(results) > 0:
            num_angles = len(results)
            
            # Setup figures (Row 1: Main/upsampled, Row 2: Downscaled/Native if upsampled)
            rows = 2 if factor > 1 else 1
            fig, axes = plt.subplots(rows, num_angles, figsize=(5 * num_angles, 5 * rows), squeeze=False)

            for col_idx, (name, epsf) in enumerate(results.items()):
                # Row 0: Full calculated resolution
                ax_top = axes[0, col_idx]
                im_top = ax_top.imshow(epsf, origin="lower", norm=AsinhNorm(1e-6, 0), cmap="magma")
                plt.colorbar(im_top, ax=ax_top, shrink=0.8)
                title_suffix = f" (x{factor} Sampled)" if factor > 1 else ""
                ax_top.set_title(f"{name}{title_suffix}")

                # Row 1: Downscaled Native (only if factor > 1)
                if factor > 1:
                    ax_bot = axes[1, col_idx]
                    epsf_red = block_reduce(epsf, factor)
                    im_bot = ax_bot.imshow(epsf_red, origin="lower", norm=AsinhNorm(1e-5, 0), cmap="magma")
                    plt.colorbar(im_bot, ax=ax_bot, shrink=0.8)
                    ax_bot.set_title(f"{name} (Native Res)")

            plt.tight_layout()
            plt.show()

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Generated {len(results)} ePSFs")
            print(f"{'=' * 60}")

        return results

    def get_fit_statistics(self, 
                           magnitude_field: str = "MAG_APER_1",
                           qFit_region_size = 5): #5px x 5px crop, as in Jay Anderson

        """
        Returns magnitudes, quality-of-fit and sharpness arrays
        """

        if self.epsf_angle0 is None:
            raise ValueError("No ePSF generated"
                            "(call build_multi_angle_epsf first)")

        if qFit_region_size % 2 == 0:
            raise ValueError("qFit_cropSize must be an odd number")

        print("Extracting the cutout for each star")
        self.cutouts.extract_all_cutouts(magnitude_field, extract_invalid_stars=True)
        self.cutouts.subtract_background_all()

        fwhm = self.compute_fwhm(self.epsf_angle0)
        print("FWHM", fwhm)

        print("Fitting...")
        q_fits = []
        C_fits = []
        mags = []
        scales = []

        for star_idx in tqdm(range(len(self.cutouts.stars))):
            
            shift_y, shift_x, (data, model) = self._fit_shift_to_original(
                                            star_idx,
                                            self.epsf_angle0,
                                            fwhm,
                                            fwhm_fraction=max(3, (qFit_region_size + self.DEFAULT_MAX_SHIFT)*self.cutouts.upsample_factor/fwhm) # force min size of residuals
                                        )

            residual = data - model

            cy = int(round(center_index(data.shape[0]) - shift_y))
            cx = int(round(center_index(data.shape[1]) - shift_x))

            cy_clamped = max(0, min(cy, data.shape[0] - 1))
            cx_clamped = max(0, min(cx, data.shape[1] - 1))
            C_fit = residual[cy_clamped, cx_clamped] / self._fit_amplitudes[star_idx]

            half_crop = qFit_region_size // 2
            y_min = cy - half_crop
            x_min = cx - half_crop

            y1 = max(0, min(y_min, data.shape[0] - qFit_region_size))
            y2 = y1 + qFit_region_size
            x1 = max(0, min(x_min, data.shape[1] - qFit_region_size))
            x2 = x1 + qFit_region_size

            res_qfit_cutout  = residual[y1:y2, x1:x2]
            data_qfit_cutout = data[y1:y2, x1:x2]
            model_qfit_cutout = model[y1:y2, x1:x2] 

            q_fit = np.sum(np.abs(res_qfit_cutout)) / self._fit_amplitudes[star_idx]

            # --- PLOT ---
            
            """plt.style.use("article_style")
            plt.figure(figsize=(10,2))
            plt.suptitle(rf"$Q_f = {q_fit:.3f}$")
            
            plt.subplot(131)
            plt.imshow(data_qfit_cutout, origin="lower", cmap="magma", vmin=0, vmax=np.max(data_qfit_cutout))
            plt.colorbar()
            
            plt.subplot(132)
            # Usiamo direttamente il ritaglio del modello (model_qfit_cutout)
            plt.imshow(model_qfit_cutout, origin="lower", cmap="magma", vmin=0, vmax=np.max(data_qfit_cutout))
            plt.colorbar()
            
            plt.subplot(133)
            m = np.max(np.abs(res_qfit_cutout))
            plt.imshow(res_qfit_cutout, origin="lower", cmap="coolwarm", vmin=-m, vmax=m)
            plt.colorbar()
            
            plt.show()"""
            

            # 5. Salvataggio delle metriche
            q_fits.append(q_fit)
            C_fits.append(C_fit)

            zero_point_mag = self.cutouts.image_collection.data[self.cutouts.stars[0].image_name].mag_zeropoint
            if zero_point_mag is None:
                zero_point_mag = 0 # instrumnetal magnitude

            mags.append(-2.5 * np.log10(self._fit_amplitudes[star_idx]) + zero_point_mag)

            scales.append(self.cutouts.stars[star_idx].scale_factor)


        return mags, q_fits, C_fits

        #quality-of-fit statistic 
        #extract a 5x5 crop


    # =========================================================================
    # PLOTTING
    # =========================================================================

    def plot_convergence_results(self, figsize: Tuple[int, int] = (12, 8)) -> None:
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
        max_value = np.max(np.abs(result.shift_corrections)) * 1.1
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
        ax.set_xlim(-max_value, max_value)
        ax.set_ylim(-max_value, max_value)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Plot 4: Correction magnitude histogram
        ax = axes[1, 1]
        for i, corrections in enumerate(result.shift_corrections):
            magnitudes = np.sqrt(corrections[:, 0] ** 2 + corrections[:, 1] ** 2)
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