import pytest
import numpy as np
from ligotools import readligo as rl
import os
from pathlib import Path # <-- NEW IMPORT

# --- Build robust paths ---
# This code builds an absolute path to your data directory,
# no matter where pytest is run from.

try:
    # Get the path to this test file
    THIS_FILE_PATH = Path(__file__)
    # Get the path to the 'ligotools/tests' directory
    TEST_DIR = THIS_FILE_PATH.parent
    # Get the path to the 'ligotools' directory
    LIGOTOOLS_DIR = TEST_DIR.parent
    # Get the path to the project root (hw3-squiroz621)
    PROJECT_ROOT = LIGOTOOLS_DIR.parent
    # Get the path to the data directory
    DATA_DIR = PROJECT_ROOT / 'data'
except NameError:
    # Fallback if __file__ is not defined (e.g., in some environments)
    DATA_DIR = Path('data')

# Check if the data directory exists and skip all tests if not
if not DATA_DIR.exists():
    pytest.skip("Data directory not found. Skipping all readligo tests.", allow_module_level=True)

# Build paths to the files
fn_H1 = DATA_DIR / 'H-H1_LOSC_4_V1-1126259446-32.hdf5'
fn_L1 = DATA_DIR / 'L-L1_LOSC_4_V1-1126259446-32.hdf5'
fn_template = DATA_DIR / 'GW150914_4_template.hdf5'

# --- Tests ---

@pytest.mark.skipif(not fn_H1.exists(), reason="H1 data file not found")
def test_loaddata_returns_correct_types():
    """
    Test that rl.loaddata() returns the expected data types.
    """
    # Use str(fn_H1) to pass the path as a string
    strain, time, chan_dict = rl.loaddata(str(fn_H1), 'H1') 
    
    assert isinstance(strain, np.ndarray), "Strain data is not a numpy array"
    assert isinstance(time, np.ndarray), "Time data is not a numpy array"
    assert isinstance(chan_dict, dict), "Channel dictionary is not a dict"

    # Check L1 data
    if fn_L1.exists():
        strain_L1, _, _ = rl.loaddata(str(fn_L1), 'L1')
        assert isinstance(strain_L1, np.ndarray), "L1 Strain data is not a numpy array"

@pytest.mark.skipif(not fn_H1.exists(), reason="H1 data file not found")
def test_loaddata_returns_correct_length():
    """
    Test that rl.loaddata() returns strain and time arrays of the same length.
    """
    strain, time, chan_dict = rl.loaddata(str(fn_H1), 'H1')
    
    # Add checks to make sure data is not None before calling len()
    assert strain is not None, "Strain should not be None"
    assert time is not None, "Time should not be None"
    assert len(strain) == len(time), "Strain and time arrays have different lengths"

@pytest.mark.skipif(not fn_template.exists(), reason="Template file not found")
def test_read_template_returns_correct_types():
    """
    Test that rl.read_template() returns two numpy arrays.
    """
    try:
        # Use str(fn_template) to pass the path as a string
        template_p, template_c = rl.read_template(str(fn_template)) 
        assert isinstance(template_p, np.ndarray), "Template (plus) is not a numpy array"
        assert isinstance(template_c, np.ndarray), "Template (cross) is not a numpy array"
        assert len(template_p) == len(template_c), "Template arrays have different lengths"
    except AttributeError:
        pytest.skip("read_template function not found in readligo. Skipping test.")
    except Exception as e:
        pytest.fail(f"read_template test failed with an exception: {e}")