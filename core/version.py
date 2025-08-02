"""
Version management for Quant Trading Service
"""

import os

def get_version():
    """Get the current version from VERSION file"""
    try:
        with open('VERSION', 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return "1.0.0"  # Default version

def get_version_info():
    """Get detailed version information"""
    version = get_version()
    return {
        'version': version,
        'major': int(version.split('.')[0]),
        'minor': int(version.split('.')[1]),
        'patch': int(version.split('.')[2]),
        'full_version': f"Quant Trading Service v{version}"
    }

# Current version
__version__ = get_version() 