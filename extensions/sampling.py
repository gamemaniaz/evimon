import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, Type, TypeVar

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
from tqdm import tqdm

from extensions.sampling_accumulators import SampleAccumulator


@dataclass
class SampledReportResults:
    population_report: ReportBase
    sample_report_result: Optional[dict] = None
    is_sampled: bool = False


# TODO: extract sampling strategy


def generate_sampled_report(
    *,
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    ReportClass: Type[ReportBase],
    report_class_params: dict,
    accumulator: SampleAccumulator,
    num_samples: int = 1000,
    randbits_size: int = 32,
    random_seed: Optional[int] = None,
) -> SampledReportResults:
    random.seed(random_seed)

    population_report = ReportClass(**report_class_params)
    population_report.run(
        reference_data=reference_data,
        current_data=current_data,
    )

    sample_size = len(reference_data)

    if sample_size >= len(current_data):
        return SampledReportResults(population_report)

    sampled_accum_result = None

    for _ in tqdm(range(num_samples), desc="Running samples"):
        random_state = random.getrandbits(randbits_size)
        current_data_sample = current_data.sample(
            n=sample_size,
            random_state=random_state,
        )
        sample_report: ReportBase = ReportClass(**report_class_params)
        sample_report.run(
            reference_data=reference_data,
            current_data=current_data_sample,
        )
        sampled_accum_result = accumulator.accumulate(
            sampled_accum_result,
            sample_report.as_dict(),
        )

    return SampledReportResults(population_report, sampled_accum_result, True)
