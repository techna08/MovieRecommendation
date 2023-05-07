import sys
import os
import logging
from io import StringIO

sys.path.append(os.path.dirname(__file__) + "/../../")
import online_eval
from online_eval.detect_data_drift import *

def test_check_data_drift():
    # Create a logger and a log buffer
    logger = logging.getLogger()
    log_buffer = StringIO()

    # Add the buffer to the logger's handlers
    handler = logging.StreamHandler(log_buffer)
    logger.addHandler(handler)
    # Define some sample data for testing
    metrics = {
        'rating': {
            'daily': {
                '2022-03-01': {'mean_rating': 3.5},
                '2022-03-02': {'mean_rating': 4.0},
                '2022-03-03': {'mean_rating': 3.8},
                '2022-03-04': {'mean_rating': 4.2},
                '2022-03-05': {'mean_rating': 4.1},
                '2022-03-06': {'mean_rating': 2.9},
                '2022-03-07': {'mean_rating': 3.7}
            }
        }
    }
    threshold = 0.005
    
    # Test case 1: No data drift detected
    result = check_data_drift(threshold, metrics)
    assert (not result)

    # Test case 2: Data drift detected
    metrics['rating']['daily']['2022-03-08'] = {'mean_rating': 2.0}
    result = check_data_drift(threshold, metrics)

    assert "Mean movie rating on" in log_buffer.getvalue()
    assert result



