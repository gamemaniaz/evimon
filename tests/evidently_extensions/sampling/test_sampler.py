import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report as EvidentlyReport
from pandas.testing import assert_frame_equal

from evidently_extensions.sampling.accumulators import (
    DatasetDriftMetricAccumValue,
    DatasetDriftMetricSampleAccumulator,
)
from evidently_extensions.sampling.sampler import (
    RefCurDfType,
    SamplerStategy,
    SmallerPopulationSizeSamplerStrategy,
    generate_sampled_report,
)


def test_custom_sampler_strategy():
    reference_data = pd.DataFrame(
        [
            {"col_a": 1, "col_b": 2},
            {"col_a": 3, "col_b": 4},
        ]
    )
    current_data = pd.DataFrame(
        [
            {"col_a": -1, "col_b": -2},
            {"col_a": -3, "col_b": -4},
        ]
    )

    class CustomSamplerStategy(SamplerStategy):
        def generate_samples(
            self, reference_data: pd.DataFrame, current_data: pd.DataFrame
        ) -> RefCurDfType:
            return reference_data.head(1), current_data.head(1)

    sampler = CustomSamplerStategy()
    reference_data_sample, current_data_sample = sampler.generate_samples(
        reference_data, current_data
    )

    expected_ref_sample = pd.DataFrame([{"col_a": 1, "col_b": 2}])
    expected_curr_sample = pd.DataFrame([{"col_a": -1, "col_b": -2}])
    assert_frame_equal(reference_data_sample, expected_ref_sample)
    assert_frame_equal(current_data_sample, expected_curr_sample)


def test_smaller_population_size_sampler_strategy():
    reference_data = pd.DataFrame(
        [
            {"col_a": 1, "col_b": 2},
            {"col_a": 3, "col_b": 4},
        ]
    )
    current_data = pd.DataFrame(
        [
            {"col_a": -1, "col_b": -2},
        ]
    )

    sampler = SmallerPopulationSizeSamplerStrategy(random_seed=1234, randbits_size=32)
    reference_data_sample, current_data_sample = sampler.generate_samples(
        reference_data, current_data
    )

    expected_ref_sample = pd.DataFrame([{"col_a": 1, "col_b": 2}])
    expected_curr_sample = pd.DataFrame([{"col_a": -1, "col_b": -2}])
    assert len(reference_data_sample) == 1
    assert len(current_data_sample) == 1
    assert_frame_equal(reference_data_sample, expected_ref_sample)
    assert_frame_equal(current_data_sample, expected_curr_sample)


def test_generate_sampled_report_results():
    reference_data = pd.DataFrame(
        [
            {"col_a": 1, "col_b": 2},
            {"col_a": 3, "col_b": 4},
        ]
    )
    current_data = pd.DataFrame(
        [
            {"col_a": -1, "col_b": -2},
        ]
    )
    sampler = SmallerPopulationSizeSamplerStrategy(random_seed=1234, randbits_size=32)

    sampled_report_results = generate_sampled_report(
        reference_data=reference_data,
        current_data=current_data,
        report_type=EvidentlyReport,
        report_class_params={"metrics": [DataDriftPreset()]},
        accumulator=DatasetDriftMetricSampleAccumulator(),
        num_samples=100,
        sampling_strategy=sampler,
    )

    assert sampled_report_results.sample_report_result == {
        "drift_share": 0.5,
        "number_of_columns": 200,
        "number_of_drifted_columns": 0,
        "share_of_drifted_columns": 0.0,
        "dataset_drift": False,
    }
