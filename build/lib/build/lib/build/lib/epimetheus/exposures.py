import os
import concurrent.futures
from typing import List, Optional
import numpy as np
from astropy.table import Table, vstack

from . import config


class PositionAngles:
    """Compute and manage position angles from exposure data."""
    
    # Columns required for PA computation
    REQUIRED_COLUMNS = ["cd11", "cd12", "cd21", "cd22"]
    
    def __init__(
        self,
        exposures: Table = None,
        PAs: np.ndarray = None,
        save_filepath: str = None
    ):
        """
        Parameters
        ----------
        exposures : Table, optional
            Exposure table with CD matrix columns
        PAs : array, optional
            Pre-computed position angles
        save_filepath : str, optional
            Path for saving/loading PAs
        """
        self.save_filepath = save_filepath  # Set this FIRST

        self.exposures = exposures
        
        if PAs is not None:
            self.PAs = np.asarray(PAs)
        elif exposures is not None:
            self.PAs = self._compute_PAs(exposures)
        elif save_filepath is not None and os.path.exists(save_filepath):
            self._load_PAs()
        else:
            raise ValueError(
                "Must provide one of: exposures, PAs, or valid save_filepath"
            )
    
    def _compute_PAs(self, exposures: Table) -> np.ndarray:
        """
        Compute position angles from exposure CD matrix.
        
        Parameters
        ----------
        exposures : Table
            Table with cd11, cd12, cd21, cd22 columns
            
        Returns
        -------
        np.ndarray
            Position angles in degrees [0, 360)
        """
        # Validate columns
        missing = [c for c in self.REQUIRED_COLUMNS if c not in exposures.colnames]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        cd11 = exposures["cd11"]
        cd12 = exposures["cd12"]
        cd21 = exposures["cd21"]
        cd22 = exposures["cd22"]
        
        # Compute determinant sign (handedness of coordinate system)
        det = cd11 * cd22 - cd12 * cd21
        sign = np.sign(det)
        sign[sign == 0] = -1
        
        # Compute position angle
        PAs = np.degrees(-np.arctan2(cd21, sign * cd11)) % 360
        
        return PAs
    
    def save_PAs(self, filepath: str = None):
        """
        Save position angles to file.
        
        Parameters
        ----------
        filepath : str, optional
            Output path. Uses self.save_filepath if not provided.
        """
        filepath = filepath or self.save_filepath
        
        if filepath is None:
            raise ValueError("No filepath provided for saving")
        
        # Create directory if needed
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        np.savetxt(filepath, self.PAs)
    
    def _load_PAs(self, filepath: str = None):
        """Load position angles from file."""
        filepath = filepath or self.save_filepath
        
        if filepath is None:
            raise ValueError("No filepath provided for loading")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"PA file not found: {filepath}")
        
        self.PAs = np.loadtxt(filepath)
    
    def most_common_angle(self, tolerance: float = 2.0) -> float:
        """
        Find the most common position angle.
        
        Parameters
        ----------
        tolerance : float
            Tolerance in degrees for grouping angles
            
        Returns
        -------
        float
            Most common angle in degrees
        """
        unique_angles = np.unique(self.PAs)
        
        counts = np.array([
            np.sum(np.abs(self.PAs - angle) < tolerance)
            for angle in unique_angles
        ])
        
        return unique_angles[np.argmax(counts)]
    
    def __len__(self):
        return len(self.PAs)
    
    def __repr__(self):
        return (
            f"PositionAngles(n={len(self)}, "
            f"range=[{self.PAs.min():.1f}, {self.PAs.max():.1f}] deg)"
        )
    
class GrizliAssociationQuery:
    """Query and fetch association data from the Grizli cutout service."""
    
    BASE_URL = "https://grizli-cutout.herokuapp.com"
    
    # Columns needed for PA calculation
    PA_COLUMNS = ["cd11", "cd12", "cd21", "cd22"]
    
    def __init__(
        self,
        ra: float,
        dec: float,
        sep: float,
        filters: List[str],
        max_workers: int = None,
        verbose: bool = True,
        is_niriss = None,
        exclude_word = "GR"
    ):
        """
        Parameters
        ----------
        ra : float
            Right ascension in degrees
        dec : float
            Declination in degrees
        sep : float
            Search radius in arcminutes
        filters : list of str
            List of filters to query (e.g., ["F115W", "F150W"])
        max_workers : int, optional
            Maximum parallel workers. Defaults to min(cpu_count, 10).
        verbose : bool
            Print progress messages
        is_niriss : bool
            Serve to ignore filters from niriss if the user wants to select only NIRCAM one
        exclude_word : str
            If a filter contains that words, it is not considered for PA.
            By defualt it's "GR" to exclude both HST and JWST NIRCam GRISM configurations
        """
        self.ra = ra
        self.dec = dec
        self.sep = sep
        self.filters = filters
        self.verbose = verbose
        self.is_niriss = is_niriss
        self.exclude_word  = exclude_word
        
        # Set max workers
        if max_workers is None:
            max_workers = min(os.cpu_count() or 1, 10)
        self.max_workers = max_workers
        
        # Internal state
        self._assoc_table = None
        self._selection_mask = None
    
    # -------------------------------------------------------------------------
    # URLs
    # -------------------------------------------------------------------------
    
    @property
    def assoc_url(self) -> str:
        """URL for association query."""
        return (
            f"{self.BASE_URL}/assoc?"
            f"coord={self.ra},{self.dec}&arcmin={self.sep}&output=csv"
        )
    
    def exposure_url(self, assoc_name: str) -> str:
        """URL for exposure query for a given association."""
        return f"{self.BASE_URL}/exposures?associations={assoc_name}"
    
    """# -------------------------------------------------------------------------
    # Filter handling
    # -------------------------------------------------------------------------
    
    @property
    def filters_normalized(self) -> List[str]:
        #Filters with 'U' removed (UVIS normalization).
        return [f.replace("U", "") for f in self.filters]"""
    
    # -------------------------------------------------------------------------
    # Query methods
    # -------------------------------------------------------------------------
    
    def query_associations(self) -> Table:
        """
        Query associations from the Grizli service.
        
        Returns
        -------
        Table
            Association table with selection mask applied
        """
        if self._assoc_table is not None:
            return self._assoc_table[self._selection_mask]
        
        # Fetch association table
        self._assoc_table = Table.read(self.assoc_url, format='csv')
        
        if self.verbose:
            available_filters = sorted(list(set(
                self._assoc_table['filter']
            )))
            print(f"Available filters: {np.array(available_filters)}")
        
        # Build selection mask
        self._selection_mask = self._build_selection_mask()
        
        if self.verbose:
            print(
                f"Found {len(self._assoc_table)} total entries. "
                f"Valid entries: {np.sum(self._selection_mask)}"
            )
        
        return self._assoc_table[self._selection_mask]
    
    def _build_selection_mask(self) -> np.ndarray:
        """Build boolean mask for valid associations."""
        table = self._assoc_table
        n_rows = len(table)
        
        # Filter matching
        filter_mask = np.zeros(n_rows, dtype=bool)
        for filt in self.filters:
            filter_mask |= [filt in table["filter"][i] and self.exclude_word not in table["filter"][i] for i in range(n_rows)]
        
        # Exclude NIRISS
        if self.is_niriss:
            instrument_mask = table['instrument_name'] == "NIRISS"
        elif self.is_niriss == False:
            instrument_mask = table['instrument_name'] != "NIRISS"
        
        return filter_mask & instrument_mask
    
    # -------------------------------------------------------------------------
    # Exposure fetching
    # -------------------------------------------------------------------------
    
    def _fetch_single_exposure(
        self,
        assoc_name: str,
        columns: List[str]
    ) -> Optional[Table]:
        """
        Fetch exposure table for a single association.
        
        Parameters
        ----------
        assoc_name : str
            Association name
        columns : list of str
            Columns to extract
            
        Returns
        -------
        Table or None
            Exposure table subset, or None if failed
        """
        try:
            table = Table.read(self.exposure_url(assoc_name), format='csv')
            
            # Return only valid columns
            valid_cols = [c for c in columns if c in table.colnames]
            return table[valid_cols] if valid_cols else None
            
        except Exception:
            return None
    
    def fetch_exposures(
        self,
        columns: List[str] = None
    ) -> Optional[Table]:
        """
        Fetch exposure data for all matching associations.
        
        Parameters
        ----------
        columns : list of str, optional
            Columns to extract. Defaults to ["cd12", "cd22"].
            
        Returns
        -------
        Table or None
            Stacked exposure table, or None if no data
        """
        if columns is None:
            columns = ["cd11","cd12", "cd21", "cd22"]
        
        # Ensure associations are queried
        assoc_table = self.query_associations()
        assoc_names = assoc_table['assoc_name']
        
        if len(assoc_names) == 0:
            if self.verbose:
                print("No associations found matching criteria.")
            return None
        
        if self.verbose:
            print(f"Fetching exposures for {len(assoc_names)} associations...")
        
        # Parallel fetch
        tables_list = self._parallel_fetch(assoc_names, columns)
        
        if self.verbose:
            print("\nFetch complete. Stacking tables...")
        
        # Stack results
        if tables_list:
            exposures = vstack(tables_list)
            self.exposures = exposures
            return exposures
        
        if self.verbose:
            print("No exposure data returned.")
        return None
    
    def _parallel_fetch(
        self,
        assoc_names: List[str],
        columns: List[str]
    ) -> List[Table]:
        """Fetch exposures in parallel using ThreadPoolExecutor."""
        tables_list = []
        
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            
            # Submit all tasks
            future_to_name = {
                executor.submit(
                    self._fetch_single_exposure, name, columns
                ): name
                for name in assoc_names
            }
            
            # Collect results
            for i, future in enumerate(
                concurrent.futures.as_completed(future_to_name)
            ):
                if self.verbose and i % 5 == 0:
                    print(
                        f"Processed {i + 1} / {len(assoc_names)}",
                        end='\r'
                    )
                
                result = future.result()
                if result is not None and len(result) > 0:
                    tables_list.append(result)
        
        return tables_list
    

    
    
    def __repr__(self) -> str:
        return (
            f"GrizliAssociationQuery(\n"
            f"  ra={self.ra}, dec={self.dec}, sep={self.sep} arcmin\n"
            f"  filters={self.filters}\n"
            f"  max_workers={self.max_workers}\n"
            f")"
        )
    

    @classmethod
    def query_pa(
        cls,
        field_name : str,
        ra: float,
        dec: float,
        sep: float,
        filter_name: str,
        cache_dir: str = None,
        extra_columns: List[str] = None,
        force_recomputation: bool = False,
        **kwargs
    ) -> PositionAngles:
        """
        Query exposures and find the most common position angle.
        
        Parameters
        ----------
        ra, dec : float
            Coordinates in degrees
        sep : float
            Search radius in arcminutes
        filter_name : str
            Single filter to match (e.g., "F150W")
        tolerance : float
            Tolerance in degrees for grouping angles
        cache_dir : str
            Directory for caching PA files
        extra_columns : list of str, optional
            Additional columns to fetch beyond PA requirements
        **kwargs
            Additional arguments passed to __init__
            
        Returns
        -------
        float
            Most common position angle in degrees
        """
        # Build cache filepath
        if cache_dir == None:
            cache_dir = config.CACHE_DIR 
        cache_filepath = os.path.join(cache_dir, f"PA/{field_name}_{filter_name}_PAs.txt")
        
        # Check cache first
        if os.path.exists(cache_filepath) and not force_recomputation:
            pa_obj = PositionAngles(save_filepath=cache_filepath)
        else:
            # Query exposures
            columns = cls.PA_COLUMNS.copy()
            if extra_columns:
                columns.extend(extra_columns)
            
            query = cls(ra, dec, sep, [filter_name], **kwargs)
            exposures = query.fetch_exposures(columns=columns)
            
            if exposures is None or len(exposures) == 0:
                raise ValueError(
                    f"No exposures found for filter {filter_name} "
                    f"at ({ra}, {dec}) within {sep} arcmin"
                )
            
            # Compute and cache PAs
            pa_obj = PositionAngles(
                exposures=exposures,
                save_filepath=cache_filepath
            )
            pa_obj.save_PAs()
        
        return pa_obj
    

    
