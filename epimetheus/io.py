from typing import Optional, List, Dict, Tuple, Iterator

from astropy.io import fits
from astropy.wcs import WCS
import os

from .common import ZeropointCalculator


class Image:
    """
    Handles loading and management of FITS image files.
    
    This class loads a science image and optionally an associated variance/weight map,
    extracting relevant metadata such as WCS information, magnitude zeropoint, and pixel scale.
    
    Parameters
    ----------
    image_path : str
        Path to the science FITS image.
    variance_path : str, optional
        Path to the variance/weight FITS image. Default is None.
    variance_type : str, optional
        Type of variance map: "var" (variance), "rms" (root mean square), 
        or "weight" (inverse variance). Default is "var".
    image_ext : int or str, optional
        FITS extension for the science image. If None, uses the primary HDU.
    variance_ext : int or str, optional
        FITS extension for the variance image. If None, uses the primary HDU.
    bunit : str or unit, optional
        If string header keyword to read, if astropy unit: brightness unit override (e.g., "MJy/sr"). If None, read from header.
    pixel_scale : float, optional
        Pixel scale override in arcseconds/pixel. If None, computed from WCS.
    wavelength : float, optional
        Wavelength override in microns for zeropoint calculation.
    best_PA : float, optional
        Best position angle in degrees for this image. Default is None.
    tolerance_pixel_scale : float, optional
        Tolerance for rounding pixel scale to avoid floating point issues.
        Default is 1.e-5.
    
    Attributes
    ----------
    image_path : str
        Path to the science image.
    variance_path : str or None
        Path to the variance image.
    variance_type : str
        Type of variance map.
    image_data : np.ndarray
        Science image data array.
    image_header : astropy.io.fits.Header
        Science image FITS header.
    var_data : np.ndarray or None
        Variance image data array (if provided).
    var_header : astropy.io.fits.Header or None
        Variance image FITS header (if provided).
    wcs : astropy.wcs.WCS
        World Coordinate System object.
    has_variance : bool
        Whether a variance map is available.
    mag_zeropoint : float
        AB magnitude zeropoint.
    pixel_scale : float
        Pixel scale in arcseconds/pixel.
    best_PA : float or None
        Best position angle in degrees.
    
    Raises
    ------
    ValueError
        If variance_type is not one of "var", "rms", or "weight".
    
    Examples
    --------
    >>> img = Image(
    ...     image_path="science.fits",
    ...     variance_path="variance.fits",
    ...     variance_type="var"
    ... )
    >>> print(f"Pixel scale: {img.pixel_scale} arcsec/px")
    >>> print(f"Zeropoint: {img.mag_zeropoint} mag")
    """
    
    ALLOWED_VARIANCE_TYPES = ["var", "rms", "weight"]
    
    def __init__(self,
                 image_path: str,
                 variance_path: Optional[str] = None,
                 variance_type: str = "var",
                 image_ext: Optional[int | str] = None,
                 variance_ext: Optional[int | str] = None,
                 bunit: Optional[str] = None,
                 pixel_scale: Optional[float] = None,
                 wavelength: Optional[float] = None,
                 best_PA: Optional[float] = None,
                 tolerance_pixel_scale: float = 1.e-5):
        
        # Validate variance type
        if variance_type not in self.ALLOWED_VARIANCE_TYPES:
            raise ValueError(
                f"Invalid variance_type: '{variance_type}'. "
                f"Allowed values are: {self.ALLOWED_VARIANCE_TYPES}"
            )
        
        # Store paths and configuration
        self.image_path = image_path
        self.variance_path = variance_path
        self.variance_type = variance_type
        self.best_PA = best_PA
        
        # Load FITS data
        self._load_image(image_ext, variance_ext)
        
        # Calculate zeropoint and pixel scale
        zp_calculator = ZeropointCalculator(
            self.image_header,
            bunit=bunit,
            pixel_scale=pixel_scale,
            wavelength=wavelength
        )
        
        self.mag_zeropoint = zp_calculator.zeropoint
        
        # Round pixel scale to avoid floating point precision issues
        unit = 1.
        raw_pixel_scale = zp_calculator.pixel_scale
        if hasattr(raw_pixel_scale, 'unit'):
            unit = raw_pixel_scale.unit
            raw_pixel_scale = raw_pixel_scale.value
        self.pixel_scale = round(raw_pixel_scale / tolerance_pixel_scale) * tolerance_pixel_scale * unit
    
    def _load_image(self, 
                    image_ext: Optional[int | str], 
                    variance_ext: Optional[int | str]) -> None:
        """
        Load science and variance FITS images.
        
        Parameters
        ----------
        image_ext : int, str, or None
            FITS extension for science image.
        variance_ext : int, str, or None
            FITS extension for variance image.
        """
        # Load science image
        if image_ext is None:
            self.image_data, self.image_header = fits.getdata(
                self.image_path, 
                header=True
            )
        else:
            self.image_data, self.image_header = fits.getdata(
                self.image_path, 
                image_ext, 
                header=True
            )
        
        # Extract WCS from header
        self.wcs = WCS(self.image_header)
        
        # Load variance image if provided
        if self.variance_path is not None:
            if variance_ext is None:
                self.var_data, self.var_header = fits.getdata(
                    self.variance_path, 
                    header=True
                )
            else:
                self.var_data, self.var_header = fits.getdata(
                    self.variance_path, 
                    variance_ext, 
                    header=True
                )
            self.has_variance = True
        else:
            self.var_data = None
            self.var_header = None
            self.has_variance = False
    
    def update_best_PA(self, best_PA: float) -> None:
        """
        Update the best position angle for this image.
        
        Parameters
        ----------
        best_PA : float
            New position angle in degrees.
        """
        self.best_PA = best_PA
    
    def __repr__(self) -> str:
        """Return string representation of the Image."""
        return (
            f"Image('{self.image_path}', "
            f"pixel_scale={self.pixel_scale:.4f}, "
            f"has_variance={self.has_variance})"
        )


class ImagesCollection:
    """
    A collection of Image objects for batch processing.
    
    This class manages multiple Image objects, providing dictionary-like access
    by image name and methods to add new images to the collection.
    
    Parameters
    ----------
    images : list of Image, optional
        List of Image objects to initialise the collection. Default is empty list.
    image_names : list of str, optional
        List of names/identifiers for each image. Must have the same length as images.
        Default is empty list.
    
    Attributes
    ----------
    image_names : list of str
        List of image identifiers.
    n_images : int
        Number of images in the collection.
    data : dict
        Dictionary mapping image names to Image objects.
    
    Raises
    ------
    ValueError
        If the number of images does not match the number of image names.
    
    Examples
    --------
    >>> # Create collection from existing images
    >>> img1 = Image("field1.fits")
    >>> img2 = Image("field2.fits")
    >>> collection = ImagesCollection(
    ...     images=[img1, img2],
    ...     image_names=["field1", "field2"]
    ... )
    
    >>> # Add images incrementally
    >>> collection = ImagesCollection()
    >>> collection.add_image(Image("field3.fits"), "field3")
    
    >>> # Access images by name
    >>> field1_data = collection.data["field1"].image_data
    >>> field1_wcs = collection["field1"].wcs
    """
    
    def __init__(self,
                 images: Optional[List[Image]] = None,
                 image_names: Optional[List[str]] = None):
        
        # Handle default mutable arguments
        if images is None:
            images = []
        if image_names is None:
            image_names = []
        
        # Validate input lengths
        if len(images) != len(image_names):
            raise ValueError(
                f"Number of images ({len(images)}) must match "
                f"number of image names ({len(image_names)})"
            )
        
        # Store image names and count
        self.image_names = list(image_names)
        self.n_images = len(self.image_names)
        
        # Create dictionary mapping names to Image objects
        self.data: Dict[str, Image] = {
            name: image for name, image in zip(self.image_names, images)
        }
    
    def add_image(self, image: Image, image_name: str) -> None:
        """
        Add a new image to the collection.
        
        Parameters
        ----------
        image : Image
            Image object to add.
        image_name : str
            Identifier for the image.
        
        Raises
        ------
        ValueError
            If an image with the same name already exists.
        """
        if image_name in self.data:
            raise ValueError(
                f"Image with name '{image_name}' already exists in collection. "
                f"Use a different name or remove the existing image first."
            )
        
        self.data[image_name] = image
        self.image_names.append(image_name)
        self.n_images += 1
    
    def remove_image(self, image_name: str) -> None:
        """
        Remove an image from the collection.
        
        Parameters
        ----------
        image_name : str
            Identifier of the image to remove.
        
        Raises
        ------
        KeyError
            If no image with the given name exists.
        """
        if image_name not in self.data:
            raise KeyError(f"No image with name '{image_name}' in collection.")
        
        del self.data[image_name]
        self.image_names.remove(image_name)
        self.n_images -= 1
    
    def __getitem__(self, image_name: str) -> "Image":
        """Return the image associated with `image_name`."""
        return self.data[image_name]

    def __contains__(self, image_name: str) -> bool:
        """Return whether `image_name` exists in the collection."""
        return image_name in self.data

    def __len__(self) -> int:
        """Return the number of images in the collection."""
        return len(self.data)

    def __iter__(self) -> Iterator[str]:
        """Iterate over image names."""
        return iter(self.data)

    def items(self) -> Iterator[Tuple[str, "Image"]]:
        """Return an iterator over `(image_name, image)` pairs."""
        return self.data.items()

    def keys(self):
        """Return the image names."""
        return self.data.keys()

    def values(self):
        """Return the stored `Image` objects."""
        return self.data.values()

    def summary_dict(self) -> Dict[str, Dict[str, object]]:
        """
        Build a summary dictionary for all images in the collection.

        Returns
        -------
        dict
            Nested dictionary with image names as top-level keys and summary
            properties as values.
        """
        summary = {}

        for name, im in self.data.items():
            image_path = getattr(im, "image_path", None)
            summary[name] = {
                "image path": os.path.basename(image_path) if image_path else None,
                "AB zeropoint": getattr(im, "mag_zeropoint", None),
                "pixel scale": getattr(im, "pixel_scale", None),
                "variance type": getattr(im, "variance_type", None)
            }

        return summary

    def _format_table(self) -> str:
        """
        Return a plain-text table summarising the collection.

        The table uses:
        - columns: image names
        - rows: selected image properties

        Returns
        -------
        str
            Formatted text table.
        """
        if not self.data:
            return "ImagesCollection(empty)"

        summary = self.summary_dict()
        field_names = list(summary.keys())
        row_names = [
            "image path",
            "AB zeropoint",
            "pixel scale",
            "variance type"
        ]

        # Build string matrix
        table = []
        header = ["property"] + field_names
        table.append(header)

        for row_name in row_names:
            row = [row_name]
            for field in field_names:
                value = summary[field][row_name]

                if isinstance(value, float):
                    if row_name == "AB zeropoint":
                        value_str = f"{value:.2f}"
                    elif row_name == "pixel scale":
                        value_str = f"{value:.5f}"
                    else:
                        value_str = f"{value:.3f}"
                else:
                    value_str = str(value)

                row.append(value_str)
            table.append(row)

        # Compute column widths
        widths = [max(len(str(row[i])) for row in table) for i in range(len(table[0]))]

        # Format rows
        lines = []
        for i, row in enumerate(table):
            line = " | ".join(str(cell).ljust(widths[j]) for j, cell in enumerate(row))
            lines.append(line)
            if i == 0:
                lines.append("-+-".join("-" * w for w in widths))

        return "\n".join(lines) + "\n"

    def __repr__(self) -> str:
        """Return a readable summary table of the image collection."""
        return self._format_table()

    __str__ = __repr__