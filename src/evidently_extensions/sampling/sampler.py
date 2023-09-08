import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Type, TypeAlias

import pandas as pd
from evidently.suite.base_suite import ReportBase
from tqdm import tqdm

from evidently_extensions.sampling.accumulators import SampleAccumulator

RefCurDfType: TypeAlias = tuple[pd.DataFrame, pd.DataFrame]


class SamplerStategy(ABC):
    @abstractmethod
    def generate_samples(
        self, reference_data: pd.DataFrame, current_data: pd.DataFrame
    ) -> RefCurDfType:
        return None, None


class SmallerPopulationSizeSamplerStrategy(SamplerStategy):
    def __init__(self, random_seed: int = None, randbits_size: int = 32) -> None:
        self.random_seed = random_seed
        self.randbits_size = randbits_size
        random.seed(self.random_seed)

    def generate_samples(
        self, reference_data: pd.DataFrame, current_data: pd.DataFrame
    ) -> RefCurDfType:
        random_state = random.getrandbits(self.randbits_size)
        sample_size = min(len(reference_data), len(current_data))
        reference_data_sample = reference_data.sample(
            n=sample_size,
            random_state=random_state,
        ).reset_index(drop=True)
        current_data_sample = current_data.sample(
            n=sample_size,
            random_state=random_state,
        ).reset_index(drop=True)
        return reference_data_sample, current_data_sample


class FixedSizeSamplerStrategy(SamplerStategy):
    def __init__(
        self, sample_size: int, random_seed: int = None, randbits_size: int = 32
    ) -> None:
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.randbits_size = randbits_size
        random.seed(self.random_seed)

    def generate_samples(
        self, reference_data: pd.DataFrame, current_data: pd.DataFrame
    ) -> RefCurDfType:
        random_state = random.getrandbits(self.randbits_size)
        reference_data_sample = reference_data.sample(
            n=self.sample_size,
            random_state=random_state,
        ).reset_index(drop=True)
        current_data_sample = current_data.sample(
            n=self.sample_size,
            random_state=random_state,
        ).reset_index(drop=True)
        return reference_data_sample, current_data_sample


class SmallerPopulationSizeStratifiedSamplerStrategy(SamplerStategy):
    def generate_samples(
        self, reference_data: pd.DataFrame, current_data: pd.DataFrame
    ) -> RefCurDfType:
        # TODO: add impl
        raise NotImplementedError()


class FixedSizeStratifiedSamplerStrategy(SamplerStategy):
    def generate_samples(
        self, reference_data: pd.DataFrame, current_data: pd.DataFrame
    ) -> RefCurDfType:
        # TODO: add impl
        raise NotImplementedError()


@dataclass
class SampledReportResults:
    population_report: ReportBase
    sample_report_result: Optional[dict] = None


def generate_sampled_report(
    *,
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    report_type: Type[ReportBase],
    report_class_params: dict,
    accumulator: SampleAccumulator,
    num_samples: int,
    sampling_strategy: SamplerStategy,
) -> SampledReportResults:
    population_report = report_type(**report_class_params)
    population_report.run(
        reference_data=reference_data,
        current_data=current_data,
    )

    sampled_accum_result = None
    for _ in tqdm(range(num_samples), desc="running samples..."):
        ref_sample, cur_sample = sampling_strategy.generate_samples(
            reference_data, current_data
        )
        sample_report: ReportBase = report_type(**report_class_params)
        sample_report.run(reference_data=ref_sample, current_data=cur_sample)
        sampled_accum_result = accumulator.accumulate(sample_report.as_dict())

    return SampledReportResults(population_report, sampled_accum_result)
