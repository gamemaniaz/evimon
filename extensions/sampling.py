import json
import random
from dataclasses import dataclass
from typing import Any, Callable, Optional, Type

import pandas as pd
from evidently.base_metric import Metric
from evidently.metric_preset.metric_preset import MetricPreset
from evidently.metrics import DatasetDriftMetric
from evidently.report import Report
from evidently.suite.base_suite import ReportBase
from evidently.test_preset import DataDriftTestPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestAllColumnsShareOfMissingValues
from evidently.utils.generators import BaseGenerator
from rich.progress import track


class AccumulatorException(Exception):
    """General accumulator exception wrapper"""


@dataclass
class SampledReportResults:
    population_report: ReportBase
    sample_report_result: Optional[dict] = None
    is_sampled: bool = False


def generate_sampled_report(
    *,
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    ReportClass: Type[ReportBase],
    report_class_params: dict,
    sample_output_accumulator: Callable,
    num_samples: int = 1000,
    randbits_size: int = 32,
    random_seed: Optional[int] = None,
) -> SampledReportResults:
    random.seed(random_seed)

    population_report = ReportClass(**report_class_params)
    population_report.run(reference_data=reference_data, current_data=current_data)

    sample_size = len(reference_data)

    if sample_size >= len(current_data):
        # TODO: add non-sampling logic
        return SampledReportResults(population_report)

    sampled_accum_result = None

    for _ in track(range(num_samples)):
        random_state = random.getrandbits(randbits_size)
        current_data_sample = current_data.sample(
            sample_size, random_state=random_state
        )
        sample_report: ReportBase = ReportClass(**report_class_params)
        sample_report.run(
            reference_data=reference_data, current_data=current_data_sample
        )
        sampled_accum_result = sample_output_accumulator(
            sample_report.as_dict(), sampled_accum_result
        )

    return SampledReportResults(population_report, sampled_accum_result, True)
