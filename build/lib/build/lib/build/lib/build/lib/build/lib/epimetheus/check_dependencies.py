import shutil
import subprocess
import warnings

def check_sextractor():
    """
    Check if SExtractor is installed and accessible.
    Returns the name and path to the executable or None.
    """
    # SExtractor can be called as 'sex', 'sextractor', or 'source-extractor'
    possible_names = ['sex', 'sextractor', 'source-extractor']
    
    for name in possible_names:
        path = shutil.which(name)
        if path is not None:
            return name, path
    
    return None, None


def get_sextractor_version(executable):
    """Get the version of SExtractor."""
    try:
        result = subprocess.run(
            [executable, '--version'], 
            capture_output=True, 
            text=True
        )
        return result.stdout.strip() or result.stderr.strip()
    except Exception:
        return "unknown"


def require_sextractor():
    """
    Check for SExtractor and raise an error with installation instructions if not found.
    """
    sex_name , sex_path = check_sextractor()
    
    if sex_path is None:
        install_instructions = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                     SExtractor is required but not found!                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Please install SExtractor using one of the following methods:                ║
║                                                                              ║
║ Ubuntu/Debian:                                                               ║
║   sudo apt-get install sextractor                                            ║
║                                                                              ║
║ macOS (Homebrew):                                                            ║
║   brew install sextractor                                                    ║
║                                                                              ║
║ Conda (any platform):                                                        ║
║   conda install -c conda-forge astromatic-source-extractor                   ║
║                                                                              ║
║ From source:                                                                 ║
║   https://github.com/astromatic/sextractor                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

        warnings.warn(
       	    install_instructions,
            UserWarning
    	)
    
    return sex_name, sex_path


def check_all_dependencies(verbose=True):
    """
    Check all external (non-Python) dependencies.
    """
    status = {}
    
    # Check SExtractor
    sex_name , sex_path = require_sextractor()
    if sex_path:
        version = get_sextractor_version(sex_path)
        status['sextractor'] = {'installed': True, 'name':sex_name, 'path': sex_path, 'version': version}
        if verbose:
            print(f"SExtractor found: {sex_path} ({version})")
    else:
        status['sextractor'] = {'installed': False, 'path': None, 'version': None}
        if verbose:
            print("SExtractor not found")
    
    return status