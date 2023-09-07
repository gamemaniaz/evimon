from typing import Any

import pytest

from evidently_extensions.sampling.accumulators import (
    DatasetDriftMetricSampleAccumulator,
    SampleAccumulator,
)
from evidently_extensions.sampling.transformers import TransformerException


def test_custom_accumulator():
    CustomAccumValue = dict[str, int]

    class CustomAccumulator(SampleAccumulator[CustomAccumValue]):
        def __init__(self):
            self.accum_value: CustomAccumValue = {"value": 0}

        def accumulate(self, new_input: Any) -> CustomAccumValue:
            self.accum_value["value"] += new_input
            return self.accum_value

    accumulator = CustomAccumulator()

    assert accumulator.accum_value == {"value": 0}
    assert accumulator.accumulate(1) == {"value": 1}
    assert accumulator.accumulate(2) == {"value": 3}
    assert accumulator.accumulate(3) == {"value": 6}


def test_dataset_drift_metric_sample_accumulator():
    accumulator = DatasetDriftMetricSampleAccumulator()

    assert accumulator.accum_value == {
        "drift_share": 0.0,
        "number_of_columns": 0,
        "number_of_drifted_columns": 0,
        "share_of_drifted_columns": 0.0,
        "dataset_drift": False,
    }
    assert accumulator.accumulate(
        {
            "metrics": [
                {
                    "metric": "DatasetDriftMetric",
                    "result": {
                        "drift_share": 0.5,
                        "number_of_columns": 10,
                        "number_of_drifted_columns": 4,
                        "share_of_drifted_columns": 0.4,
                        "dataset_drift": False,
                    },
                }
            ]
        }
    ) == {
        "drift_share": 0.5,
        "number_of_columns": 10,
        "number_of_drifted_columns": 4,
        "share_of_drifted_columns": 0.4,
        "dataset_drift": False,
    }
    assert accumulator.accumulate(
        {
            "metrics": [
                {
                    "metric": "DatasetDriftMetric",
                    "result": {
                        "drift_share": 0.5,
                        "number_of_columns": 10,
                        "number_of_drifted_columns": 6,
                        "share_of_drifted_columns": 0.6,
                        "dataset_drift": True,
                    },
                }
            ]
        }
    ) == {
        "drift_share": 0.5,
        "number_of_columns": 20,
        "number_of_drifted_columns": 10,
        "share_of_drifted_columns": 0.5,
        "dataset_drift": True,
    }

    with pytest.raises(TransformerException):
        accumulator.accumulate({"some_unknown_field": 20})
