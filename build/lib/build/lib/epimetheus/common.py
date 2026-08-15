# epi/core.py
import numpy as np
from astropy import units as u

class ZeropointCalculator:
    """Calculate AB magnitude zeropoint for astronomical images."""
    
    # AB mag reference: 3631 Jy
    AB_REFERENCE_JY = 3631.0
    
    def __init__(self, header, bunit=None, pixel_scale=None, wavelength=None):
        """
        Parameters
        ----------
        header : fits.Header
            FITS header
        bunit : str, u.Unit, or None
            Unit specification (header keyword or unit)
        pixel_scale : float or Quantity, optional
            Pixel scale in arcsec/pixel. Required for /sr units.
        wavelength : float or Quantity, optional
            Effective wavelength. Required for erg/s/cm²/Å type units.
            If float, assumed to be in Angstrom.
        """
        self.header = header
        self.unit = self._parse_unit(bunit)
        self.pixel_scale = self._parse_pixel_scale(pixel_scale)
        self.wavelength = self._parse_wavelength(wavelength)
        
        # Determine unit type
        self.unit_type = self._classify_unit()
        
    def _parse_unit(self, bunit):
        """Parse unit from header keyword or direct unit."""
        if bunit is None:
            bunit = "BUNIT"
        
        if isinstance(bunit, str):
            unit_str = self.header.get(bunit)
            if unit_str is None:
                raise KeyError(f"Header keyword '{bunit}' not found")
            return u.Unit(unit_str)
        
        elif isinstance(bunit, u.UnitBase):
            return bunit
        
        raise TypeError(f"bunit must be str or Unit, got {type(bunit).__name__}")
    
    def _parse_pixel_scale(self, pixel_scale):
        """Parse pixel scale."""
        if pixel_scale is None:
            # Try header
            for key in ["CD1_1","PIXSCALE", "CDELT1"]:
                if key in self.header:
                    val = np.abs(self.header[key])
                    if key == "CDELT1":
                        return (val * u.deg).to(u.arcsec)
                    elif key == "CD1_1":
                         val = np.sqrt((self.header["CD1_1"])**2 + (self.header["CD2_1"])**2)
                         return (val * u.deg).to(u.arcsec)
                    return val * u.arcsec
            return None
        
        if isinstance(pixel_scale, u.Quantity):
            return pixel_scale.to(u.arcsec)
        return pixel_scale * u.arcsec
    
    def _parse_wavelength(self, wavelength):
        """Parse wavelength."""
        if wavelength is None:
            # Try common header keywords
            for key in ["PHOTPLAM", "WAVELENG", "LAMBDA_C", "CENWAVE"]:
                if key in self.header:
                    return self.header[key] * u.AA
            return None
        
        if isinstance(wavelength, u.Quantity):
            return wavelength.to(u.AA)
        return wavelength * u.AA
    
    def _classify_unit(self):
        """Classify the unit type."""
        # Check for frequency-based flux density (Jy, erg/s/cm²/Hz)
        try:
            self.unit.to(u.Jy)
            return "f_nu"
        except u.UnitConversionError:
            pass
        
        # Check for frequency-based surface brightness (/sr)
        try:
            (self.unit * u.sr).to(u.Jy)
            return "f_nu_per_sr"
        except u.UnitConversionError:
            pass
        
        # Check for wavelength-based flux density (erg/s/cm²/Å)
        try:
            self.unit.to(u.erg / u.s / u.cm**2 / u.AA)
            return "f_lambda"
        except u.UnitConversionError:
            pass
        
        # Check for wavelength-based surface brightness
        try:
            (self.unit * u.sr).to(u.erg / u.s / u.cm**2 / u.AA)
            return "f_lambda_per_sr"
        except u.UnitConversionError:
            pass
        
        raise ValueError(f"Cannot classify unit: {self.unit}")
    
    @property
    def pixel_solid_angle(self):
        """Pixel solid angle in steradians."""
        if self.pixel_scale is None:
            raise ValueError(
                "pixel_scale required for surface brightness units. "
                "Provide pixel_scale or add PIXSCALE/CDELT1 to header."
            )
        return (self.pixel_scale.to(u.rad).value ** 2) * u.sr
    
    def _f_lambda_to_f_nu(self, f_lambda):
        """
        Convert f_λ to f_ν.
        
        f_ν = f_λ * λ² / c
        """
        if self.wavelength is None:
            raise ValueError(
                "wavelength required for f_λ units (erg/s/cm²/Å). "
                "Provide wavelength parameter or add PHOTPLAM/WAVELENG to header."
            )
        
        # f_nu = f_lambda * lambda^2 / c
        c = 2.998e18 * u.AA / u.s  # speed of light in Å/s
        f_nu = f_lambda * (self.wavelength**2) / c
        
        return f_nu.to(u.erg / u.s / u.cm**2 / u.Hz)
    
    def _to_jy(self, value=1.0):
        """Convert a pixel value to Jy."""
        
        if self.unit_type == "f_nu":
            # Direct conversion to Jy
            return (value * self.unit).to(u.Jy)
        
        elif self.unit_type == "f_nu_per_sr":
            # Multiply by pixel solid angle
            flux = value * self.unit * self.pixel_solid_angle
            return flux.to(u.Jy)
        
        elif self.unit_type == "f_lambda":
            # Convert f_λ to f_ν, then to Jy
            f_lambda = value * self.unit
            f_nu = self._f_lambda_to_f_nu(f_lambda)
            return f_nu.to(u.Jy)
        
        elif self.unit_type == "f_lambda_per_sr":
            # Multiply by solid angle, then convert f_λ to f_ν
            f_lambda = value * self.unit * self.pixel_solid_angle
            f_nu = self._f_lambda_to_f_nu(f_lambda)
            return f_nu.to(u.Jy)
    
    @property
    def zeropoint(self):
        """
        Calculate AB magnitude zeropoint.
        
        Zeropoint = magnitude when pixel value = 1
        """
        flux_jy = self._to_jy(value=1.0).value
        
        # m_AB = -2.5 * log10(f_Jy / 3631)
        zp = -2.5 * np.log10(flux_jy / self.AB_REFERENCE_JY)
        
        return zp
    
    def __repr__(self):
        info = [
            f"ZeropointCalculator(",
            f"  unit = {self.unit}",
            f"  type = {self.unit_type}",
        ]
        if self.pixel_scale is not None:
            info.append(f"  pixel_scale = {self.pixel_scale:.4f}")
        if self.wavelength is not None:
            info.append(f"  wavelength = {self.wavelength:.1f}")
        info.append(f"  zeropoint = {self.zeropoint:.4f} ABmag")
        info.append(")")
        return "\n".join(info)