from abc import ABC, abstractmethod
from typing import Any, Generic, TypedDict, TypeVar

from evidently_extensions.sampling.transformers import get_metric_result, get_test_result

T = TypeVar("T")


class SampleAccumulator(ABC, Generic[T]):
    @abstractmethod
    def accumulate(self, new_input: Any) -> T:
        return


class DatasetDriftMetricAccumValue(TypedDict):
    drift_share: float
    number_of_columns: int
    number_of_drifted_columns: int
    share_of_drifted_columns: float
    dataset_drift: bool


class DatasetDriftMetricSampleAccumulator(
    SampleAccumulator[DatasetDriftMetricAccumValue]
):
    def __init__(self) -> None:
        self.accum_value = DatasetDriftMetricAccumValue(
            drift_share=0.0,
            number_of_columns=0,
            number_of_drifted_columns=0,
            share_of_drifted_columns=0.0,
            dataset_drift=False,
        )

    def accumulate(self, new_input: Any) -> DatasetDriftMetricAccumValue:
        new_input = get_metric_result(new_input, "DatasetDriftMetric")
        self.accum_value["number_of_columns"] += new_input["number_of_columns"]
        self.accum_value["number_of_drifted_columns"] += new_input[
            "number_of_drifted_columns"
        ]
        self.accum_value["drift_share"] = new_input["drift_share"]
        self.accum_value["share_of_drifted_columns"] = round(
            self.accum_value["number_of_drifted_columns"]
            / self.accum_value["number_of_columns"],
            2,
        )
        self.accum_value["dataset_drift"] = (
            self.accum_value["share_of_drifted_columns"]
            >= self.accum_value["drift_share"]
        )
        return self.accum_value


# class NumberOfDriftedColumnsSampleAccumulator(
#     SampleAccumulator[dict]
# ):
#     def __init__(self) -> None:
#         self.accum_value = dict()

#     def accumulate(self, new_input: Any) -> dict:
#         return super().accumulate(new_input)
    