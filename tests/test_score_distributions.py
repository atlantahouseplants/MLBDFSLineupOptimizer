import os
import pytest
import numpy as np
from scipy.stats import lognorm

from slate_optimizer.projection.score_distributions import fit_player_distributions, _calculate_lognorm_params

# Helper to set/unset environment variables for testing
@pytest.fixture(autouse=True)
def set_env_for_testing():
    original_env_value = os.environ.get("COMPUTED_BOOM_BUST")
    os.environ["COMPUTED_BOOM_BUST"] = "false" # Default to false for most tests
    yield
    if original_env_value is None:
        del os.environ["COMPUTED_BOOM_BUST"]
    else:
        os.environ["COMPUTED_BOOM_BUST"] = original_env_value


def test_calculate_lognorm_params_sane_inputs():
    # Test with typical mean and CV
    mu, sigma = _calculate_lognorm_params(mean=10.0, cv=0.5)
    assert isinstance(mu, float)
    assert isinstance(sigma, float)
    assert sigma > 0
    # Verify that the calculated parameters yield the original mean and CV approximately
    calculated_mean = float(np.exp(mu + sigma**2 / 2))
    calculated_cv = float(np.sqrt(np.exp(sigma**2) - 1))
    np.testing.assert_almost_equal(calculated_mean, 10.0, decimal=5)
    np.testing.assert_almost_equal(calculated_cv, 0.5, decimal=5)

    # Test with low CV (less variance)
    mu, sigma = _calculate_lognorm_params(mean=10.0, cv=0.1)
    calculated_mean = float(np.exp(mu + sigma**2 / 2))
    calculated_cv = float(np.sqrt(np.exp(sigma**2) - 1))
    np.testing.assert_almost_equal(calculated_mean, 10.0, decimal=5)
    np.testing.assert_almost_equal(calculated_cv, 0.1, decimal=5)

    # Test with high CV (more variance)
    mu, sigma = _calculate_lognorm_params(mean=10.0, cv=1.0)
    calculated_mean = float(np.exp(mu + sigma**2 / 2))
    calculated_cv = float(np.sqrt(np.exp(sigma**2) - 1))
    np.testing.assert_almost_equal(calculated_mean, 10.0, decimal=5)
    np.testing.assert_almost_equal(calculated_cv, 1.0, decimal=5)

    # Test with mean close to zero
    mu, sigma = _calculate_lognorm_params(mean=0.001, cv=0.5)
    assert isinstance(mu, float)
    assert isinstance(sigma, float)

def test_fit_player_distributions_flag_off():
    # When flag is off, should return crude values
    player_data = {'PointsFD': 20.0, 'HomeRunProbability': 0.1}
    player_type = 'batter'

    results = fit_player_distributions(player_data, player_type)

    assert results['median_fd_points'] == 20.0
    assert results['floor_fd_points'] == 20.0 * 0.8
    assert results['ceiling_fd_points'] == 20.0 * 1.2
    assert results['boom_percentage'] == 0.0
    assert results['bust_percentage'] == 0.0

def test_fit_player_distributions_flag_on_batter():
    os.environ["COMPUTED_BOOM_BUST"] = "true"
    player_data = {
        'PointsFD': 20.0,
        'HomeRunProbability': 0.15,
        'HitProbability': 0.8,
        'StolenBaseProbability': 0.05,
        'Innings': 5.0, # Dummy for pitchers, should be ignored for batters
        'Strikeouts': 5.0 # Dummy for pitchers
    }
    player_type = 'batter'

    results = fit_player_distributions(player_data, player_type)

    assert results['median_fd_points'] > results['floor_fd_points']
    assert results['ceiling_fd_points'] > results['median_fd_points']
    assert 0 <= results['boom_percentage'] <= 1
    assert 0 <= results['bust_percentage'] <= 1
    assert results['boom_percentage'] + results['bust_percentage'] <= 1.0 # Or slightly higher due to median definition

    # Test monotonic CV with HR probability
    player_data_low_hr = player_data.copy()
    player_data_low_hr['HomeRunProbability'] = 0.05
    results_low_hr = fit_player_distributions(player_data_low_hr, player_type)

    player_data_high_hr = player_data.copy()
    player_data_high_hr['HomeRunProbability'] = 0.25
    results_high_hr = fit_player_distributions(player_data_high_hr, player_type)

    # Higher HR probability should lead to higher CV (larger spread between floor/ceiling relative to median)
    # Check if the range (ceiling - floor) / median increases with HR probability
    range_low_hr = (results_low_hr['ceiling_fd_points'] - results_low_hr['floor_fd_points']) / results_low_hr['median_fd_points']
    range_high_hr = (results_high_hr['ceiling_fd_points'] - results_high_hr['floor_fd_points']) / results_high_hr['median_fd_points']
    assert range_high_hr > range_low_hr

def test_fit_player_distributions_flag_on_pitcher():
    os.environ["COMPUTED_BOOM_BUST"] = "true"
    player_data = {
        'PointsFD': 15.0,
        'HomeRunProbability': 0.0, # Dummy for batters
        'HitProbability': 0.0, # Dummy for batters
        'StolenBaseProbability': 0.0, # Dummy for batters
        'Innings': 6.0,
        'Strikeouts': 7.0,
        'BB': 2.0,
        'WinPct': 0.5
    }
    player_type = 'pitcher'

    results = fit_player_distributions(player_data, player_type)

    assert results['median_fd_points'] > results['floor_fd_points']
    assert results['ceiling_fd_points'] > results['median_fd_points']
    assert 0 <= results['boom_percentage'] <= 1
    assert 0 <= results['bust_percentage'] <= 1
    assert results['boom_percentage'] + results['bust_percentage'] <= 1.0

    # Test monotonic CV with K rate
    player_data_low_k = player_data.copy()
    player_data_low_k['Strikeouts'] = 4.0
    results_low_k = fit_player_distributions(player_data_low_k, player_type)

    player_data_high_k = player_data.copy()
    player_data_high_k['Strikeouts'] = 10.0
    results_high_k = fit_player_distributions(player_data_high_k, player_type)

    # Higher K rate should lead to higher CV
    range_low_k = (results_low_k['ceiling_fd_points'] - results_low_k['floor_fd_points']) / results_low_k['median_fd_points']
    range_high_k = (results_high_k['ceiling_fd_points'] - results_high_k['floor_fd_points']) / results_high_k['median_fd_points']
    assert range_high_k > range_low_k