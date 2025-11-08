import pytest
import numpy as np
from ligotools import readligo as rl
import os

# We need to make sure we can find the data files
# This assumes pytest is run from the root directory (hw3-squiroz621)
data_dir = 'data/'
fn_H1 = os.path.join(data_dir, 'H-H1_LOSC_4_V1-1126259446-32.hdf5')
fn_L1 = os.path.join(data_dir, 'L-L1_LOSC_4_V1-1126259446-32.hdf5')
fn_template = os.path.join(data_dir, 'GW150914_4_template.hdf5')

def test_loaddata_returns_correct_types():
    """
    Test that rl.loaddata() returns the expected data types.
    """
    # Check H1 data
    strain, time, chan_dict = rl.loaddata(fn_H1, 'H1')
    
    assert isinstance(strain, np.ndarray), "Strain data is not a numpy array"
    assert isinstance(time, np.ndarray), "Time data is not a numpy array"
    assert isinstance(chan_dict, dict), "Channel dictionary is not a dict"

    # Check L1 data
    strain_L1, time_L1, chan_dict_L1 = rl.loaddata(fn_L1, 'L1')
    assert isinstance(strain_L1, np.ndarray), "L1 Strain data is not a numpy array"

def test_loaddata_returns_correct_length():
    """
    Test that rl.loaddata() returns strain and time arrays of the same length.
    """
    strain, time, chan_dict = rl.loaddata(fn_H1, 'H1')
    
    assert len(strain) == len(time), "Strain and time arrays have different lengths"

def test_read_template_returns_correct_types():
    """
    Test that rl.read_template() returns two numpy arrays.
    """
    try:
        template_p, template_c = rl.read_template(fn_template)
        assert isinstance(template_p, np.ndarray), "Template (plus) is not a numpy array"
        assert isinstance(template_c, np.ndarray), "Template (cross) is not a numpy array"
        assert len(template_p) == len(template_c), "Template arrays have different lengths"
    except AttributeError:
        pytest.skip("read_template function not found in readligo. Skipping test.")
    except Exception as e:
        pytest.fail(f"read_template test failed with an exception: {e}")