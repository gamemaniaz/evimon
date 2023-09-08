from abc import ABC, abstractmethod
from typing import Any, Generic, TypedDict, TypeVar

from evidently_extensions.sampling.transformers import (
    get_metric_result,
    get_test_result,
    get_test_results,
)

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


class DataDriftTableAccumValue(TypedDict):
    drift_share: float
    number_of_columns: int
    number_of_drifted_columns: int
    share_of_drifted_columns: float
    dataset_drift: bool


class DataDriftTableSampleAccumulator(SampleAccumulator[DataDriftTableAccumValue]):
    def __init__(self, drift_share: float = 0.5) -> None:
        self.drift_share = drift_share
        self.accum_value = DataDriftTableAccumValue(
            drift_share=self.drift_share,
            number_of_columns=0,
            number_of_drifted_columns=0,
            share_of_drifted_columns=0.0,
            dataset_drift=False,
        )

    def accumulate(self, new_input: Any) -> DataDriftTableAccumValue:
        new_input = get_metric_result(new_input, "DataDriftTable")
        self.accum_value["number_of_columns"] += new_input["number_of_columns"]
        self.accum_value["number_of_drifted_columns"] += new_input[
            "number_of_drifted_columns"
        ]
        self.accum_value["drift_share"] = self.drift_share
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


class ShareOfDriftedColumnsAccumValue(TypedDict):
    drift_share: float
    number_of_columns: int
    number_of_drifted_columns: int
    share_of_drifted_columns: float
    dataset_drift: bool


class ShareOfDriftedColumnsSampleAccumulator(
    SampleAccumulator[ShareOfDriftedColumnsAccumValue]
):
    def __init__(self) -> None:
        self.accum_value = ShareOfDriftedColumnsAccumValue(
            drift_share=0.0,
            number_of_columns=0,
            number_of_drifted_columns=0,
            share_of_drifted_columns=0.0,
            dataset_drift=False,
        )

    def accumulate(self, new_input: Any) -> ShareOfDriftedColumnsAccumValue:
        new_input = get_test_result(new_input, "Share of Drifted Columns")
        self.accum_value["drift_share"] = new_input["condition"]["lt"]
        self.accum_value["number_of_columns"] += len(new_input["features"])
        self.accum_value["number_of_drifted_columns"] += len(
            [x for x in new_input["features"] if new_input["features"][x]["detected"]]
        )
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


class NumberOfDriftedFeaturesAccumValue(TypedDict):
    num_cols_drift_share: int
    number_of_columns: int
    number_of_drifted_columns: int
    dataset_drift: bool


class NumberOfDriftedFeaturesSampleAccumulator(
    SampleAccumulator[NumberOfDriftedFeaturesAccumValue]
):
    def __init__(self) -> None:
        self.accum_value = NumberOfDriftedFeaturesAccumValue(
            num_cols_drift_share=0,
            number_of_columns=0,
            number_of_drifted_columns=0,
            dataset_drift=False,
        )

    def accumulate(self, new_input: Any) -> NumberOfDriftedFeaturesAccumValue:
        new_input = get_test_result(new_input, "Number of Drifted Features")
        self.accum_value["num_cols_drift_share"] += new_input["condition"]["lt"]
        self.accum_value["number_of_columns"] += len(new_input["features"])
        self.accum_value["number_of_drifted_columns"] += len(
            [x for x in new_input["features"] if new_input["features"][x]["detected"]]
        )
        self.accum_value["dataset_drift"] = (
            self.accum_value["number_of_drifted_columns"]
            >= self.accum_value["num_cols_drift_share"]
        )
        return self.accum_value


class AggDriftPerColumnAccumValue(TypedDict):
    drift_share: float
    number_of_columns: int
    number_of_drifted_columns: int
    share_of_drifted_columns: float
    dataset_drift: bool


class AggDriftPerColumnSampleAccumulator(
    SampleAccumulator[AggDriftPerColumnAccumValue]
):
    def __init__(self, drift_share: float = 0.5) -> None:
        self.drift_share = drift_share
        self.accum_value = AggDriftPerColumnAccumValue(
            drift_share=self.drift_share,
            number_of_columns=0,
            number_of_drifted_columns=0,
            share_of_drifted_columns=0.0,
            dataset_drift=False,
        )

    def accumulate(self, new_input: Any) -> AggDriftPerColumnAccumValue:
        new_input = get_test_results(new_input, "Drift per Column")
        self.accum_value["drift_share"] = self.drift_share
        self.accum_value["number_of_columns"] += len(new_input)
        self.accum_value["number_of_drifted_columns"] += len(
            [x for x in new_input if x["detected"]]
        )
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
