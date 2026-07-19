"""Analysis helpers for slate review and portfolio diagnostics."""

from . import backtest
from .boom_bust_calibration import BoomBustCalibrationReport, build_boom_bust_calibration_report
from .calibration_feedback import CalibrationFeedback, build_calibration_feedback
from .exposure_tuner import ExposureTuningConfig, build_exposure_recommendations, player_overrides_text, stack_cap_maps
from .failure_attribution import FailureAttributionReport, build_failure_attribution_report
from .late_news import LateNewsReport, build_late_news_report
from .lineup_scorecard import build_lineup_scorecard
from .model_calibration import ModelCalibrationReport, build_model_calibration_report
from .optimizer_tuning import OptimizerWeightTuningReport, build_optimizer_weight_tuning_report
from .post_slate_learning import PostSlateLearningReport, build_post_slate_learning_report
from .portfolio_audit import PortfolioAuditReport, audit_report_to_csv, build_portfolio_audit
from .portfolio_frontier import PortfolioFrontierReport, build_portfolio_frontier_report
from .portfolio_stress import PortfolioStressReport, build_portfolio_stress_report
from .preset_recommender import PresetRecommendation, recommend_slate_preset
from .run_audit import RunAuditReport, build_run_audit_report
from .slate_replay import SlateReplayReport, build_slate_replay_report
from .stack_exposure import StackExposureConfig, build_stack_exposure_plan, stack_plan_weights
from .strategy_backtest import StrategyBacktestReport, build_strategy_backtest_report

__all__ = [
    "backtest",
    "BoomBustCalibrationReport",
    "CalibrationFeedback",
    "PortfolioAuditReport",
    "PortfolioFrontierReport",
    "PortfolioStressReport",
    "LateNewsReport",
    "FailureAttributionReport",
    "ModelCalibrationReport",
    "OptimizerWeightTuningReport",
    "ExposureTuningConfig",
    "PostSlateLearningReport",
    "PresetRecommendation",
    "RunAuditReport",
    "SlateReplayReport",
    "StackExposureConfig",
    "StrategyBacktestReport",
    "audit_report_to_csv",
    "build_portfolio_audit",
    "build_portfolio_frontier_report",
    "build_boom_bust_calibration_report",
    "build_portfolio_stress_report",
    "build_calibration_feedback",
    "build_failure_attribution_report",
    "build_optimizer_weight_tuning_report",
    "recommend_slate_preset",
    "build_slate_replay_report",
    "build_late_news_report",
    "build_lineup_scorecard",
    "build_model_calibration_report",
    "build_exposure_recommendations",
    "player_overrides_text",
    "stack_cap_maps",
    "build_post_slate_learning_report",
    "build_run_audit_report",
    "build_stack_exposure_plan",
    "build_strategy_backtest_report",
    "stack_plan_weights",
]
