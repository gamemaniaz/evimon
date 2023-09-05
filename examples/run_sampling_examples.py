import json
from typing import Any

import pandas as pd
from evidently.metrics import DatasetDriftMetric
from evidently.report import Report
from evidently.test_preset import DataDriftTestPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfDriftedColumns, TestShareOfDriftedColumns
from sklearn import datasets

from extensions.sampling import generate_sampled_report
from extensions.sampling_accumulators import DatasetDriftMetricSampleAccumulator

# def example_report_accumulator(accum_value: Any, new_value: Any = None) -> Any:
#     """
#     {
#         "metrics": [
#             {
#                 "metric": "DatasetDriftMetric",
#                 "result": {
#                     "drift_share": 0.5,
#                     "number_of_columns": 15,
#                     "number_of_drifted_columns": 0,
#                     "share_of_drifted_columns": 0.0,
#                     "dataset_drift": false
#                 }
#             }
#         ]
#     }
#     """
#     metric = next(
#         iter(
#             [
#                 x
#                 for x in new_value.get("metrics", {})
#                 if x.get("metric", "") == "DatasetDriftMetric"
#             ]
#         ),
#         {},
#     ).get("result", None)

#     if metric is None:
#         raise AccumulatorException(
#             f"unexpected metric schema : {json.dumps(new_value, indent=4)}"
#         )

#     if accum_value is None:
#         share_of_drifted_columns = round(
#             metric["number_of_drifted_columns"] / metric["number_of_columns"], 2
#         )
#         return {
#             "number_of_drifted_columns": metric["number_of_drifted_columns"],
#             "number_of_columns": metric["number_of_columns"],
#             "drift_share": metric["drift_share"],
#             "share_of_drifted_columns": share_of_drifted_columns,
#             "dataset_drift": share_of_drifted_columns > metric["drift_share"],
#         }

#     accum_number_of_drifted_columns = (
#         metric["number_of_drifted_columns"] + accum_value["number_of_drifted_columns"]
#     )
#     accum_number_of_columns = (
#         metric["number_of_columns"] + accum_value["number_of_columns"]
#     )
#     share_of_drifted_columns = round(
#         accum_number_of_drifted_columns / accum_number_of_columns, 2
#     )

#     return {
#         "number_of_drifted_columns": accum_number_of_drifted_columns,
#         "number_of_columns": accum_number_of_columns,
#         "drift_share": metric["drift_share"],
#         "share_of_drifted_columns": share_of_drifted_columns,
#         "dataset_drift": share_of_drifted_columns > metric["drift_share"],
#     }


def test_report_sampling():
    random_seed = 2023
    adult_data: pd.DataFrame = datasets.fetch_openml(
        name="adult",
        version=2,
        as_frame="auto",
        parser="auto",
    ).frame
    reference_data = adult_data.sample(frac=0.2, random_state=random_seed)
    current_data = adult_data.drop(reference_data.index)

    sampled_report_results = generate_sampled_report(
        reference_data=reference_data,
        current_data=current_data,
        ReportClass=Report,
        report_class_params={"metrics": [DatasetDriftMetric()]},
        accumulator=DatasetDriftMetricSampleAccumulator(),
        random_seed=random_seed,
        num_samples=100,
    )

    print(sampled_report_results.sample_report_result)


# def example_testsuite_accumulator(accum_value: Any, new_value: Any = None) -> Any:
#     metric = next(
#         iter(
#             [
#                 x
#                 for x in new_value.get("metrics", {})
#                 if x.get("metric", "") == "DatasetDriftMetric"
#             ]
#         ),
#         {},
#     ).get("result", None)

#     if metric is None:
#         raise AccumulatorException(
#             f"unexpected metric schema : {json.dumps(new_value, indent=4)}"
#         )

#     if accum_value is None:
#         share_of_drifted_columns = round(
#             metric["number_of_drifted_columns"] / metric["number_of_columns"], 2
#         )
#         return {
#             "number_of_drifted_columns": metric["number_of_drifted_columns"],
#             "number_of_columns": metric["number_of_columns"],
#             "drift_share": metric["drift_share"],
#             "share_of_drifted_columns": share_of_drifted_columns,
#             "dataset_drift": share_of_drifted_columns > metric["drift_share"],
#         }

#     accum_number_of_drifted_columns = (
#         metric["number_of_drifted_columns"] + accum_value["number_of_drifted_columns"]
#     )
#     accum_number_of_columns = (
#         metric["number_of_columns"] + accum_value["number_of_columns"]
#     )
#     share_of_drifted_columns = round(
#         accum_number_of_drifted_columns / accum_number_of_columns, 2
#     )

#     return {
#         "number_of_drifted_columns": accum_number_of_drifted_columns,
#         "number_of_columns": accum_number_of_columns,
#         "drift_share": metric["drift_share"],
#         "share_of_drifted_columns": share_of_drifted_columns,
#         "dataset_drift": share_of_drifted_columns > metric["drift_share"],
#     }


# def test_testsuite_sampling():
#     random_seed = 2023
#     adult_data: pd.DataFrame = datasets.fetch_openml(
#         name="adult",
#         version=2,
#         as_frame="auto",
#         parser="auto",
#     ).frame
#     reference_data = adult_data.sample(frac=0.2, random_state=random_seed)
#     current_data = adult_data.drop(reference_data.index)

#     sampled_report_results = generate_sampled_report(
#         reference_data=reference_data,
#         current_data=current_data,
#         ReportClass=TestSuite,
#         report_class_params={"tests": [DataDriftTestPreset()]},
#         accumulate_sampled_result=example_testsuite_accumulator,
#         random_seed=random_seed,
#     )

#     print(sampled_report_results.sample_report_result)


if __name__ == "__main__":
    test_report_sampling()
