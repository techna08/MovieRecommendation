from scipy import stats
import logging
import os, sys
sys.path.append(os.path.dirname(__file__) + "/../online_eval/")
from online_evaluate import compute_online_metrics

logging.basicConfig(
    format="%(levelname)-8s %(message)s",
    level=logging.WARNING
)
logger = logging.getLogger(__name__)

def check_data_drift(p_value_threshold = 0.005, metrics = None):
    '''
    Checks data drift in daily mean movie ratings based on a p-value threshold.

    Args:
    - p_value_threshold (float): the threshold p-value for the t-test, defaults to 0.05
    - metrics (dictionary): analyzed data

    Returns:
    - bool: True if data drift is detected, False otherwise
    '''

    if not metrics:
        metrics = compute_online_metrics()

    daily_rating_metrics = metrics['rating']['daily']
    daily_mean_ratings = []
    dates = []

    for check_date in daily_rating_metrics:
        daily_mean_ratings.append(daily_rating_metrics[check_date]['mean_rating'])
        dates.append(check_date)

    # Calculates t-statistic and p-value.
    t_statistic, p_value = stats.ttest_1samp(
                                                daily_mean_ratings[:-1], 
                                                daily_mean_ratings[-1]
                                            )

    # Determines whether there is data drift based on the p-value.
    if p_value < p_value_threshold:
        if t_statistic < 0:
            logger.warning(
                f"Mean movie rating on {check_date} is too large compared to ratings on {dates[:-1]} based on p-value {p_value_threshold}"
            )
        else:
            logger.warning(
                f"Mean movie rating on {check_date} is too small compared to ratings on {dates[:-1]} based on p-value {p_value_threshold}"
            )
        return True
    return False

if __name__ == "__main__":
    detect_data_drift()