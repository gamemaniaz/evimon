from abc import ABC, abstractmethod
from typing import Union

from extensions.data_objects import DOBase, DODatasetDriftMetric


# TODO: think about accum output schema might be different from accum inputs 


class SampleAccumulator(ABC):
    @abstractmethod
    def accumulate(
        cls,
        accum_value: Union[DOBase, dict],
        new_value: Union[DOBase, dict],
    ) -> DOBase:
        return


class AccumulatorException(Exception):
    """General accumulator exception wrapper"""


class DatasetDriftMetricSampleAccumulator(SampleAccumulator):
    def accumulate(
        self,
        accum_value: Union[DODatasetDriftMetric, dict],
        new_value: Union[DODatasetDriftMetric, dict],
    ) -> DODatasetDriftMetric:
        # TODO: think of another way to serialise when required
        if accum_value is not None and type(accum_value) is dict:
            accum_value: DODatasetDriftMetric = DODatasetDriftMetric.from_dict(
                accum_value
            )
        if new_value is not None and type(new_value) is dict:
            new_value: DODatasetDriftMetric = DODatasetDriftMetric.from_dict(new_value)
        # metric = next(
        #     iter(
        #         [
        #             x
        #             for x in new_value.get("metrics", {})
        #             if x.get("metric", "") == "DatasetDriftMetric"
        #         ]
        #     ),
        #     {},
        # ).get("result", None)

        # if metric is None:
        #     raise AccumulatorException(
        #         f"unexpected metric schema : {json.dumps(new_value, indent=4)}"
        #     )

        if accum_value is None:
            share_of_drifted_columns = round(
                new_value.number_of_drifted_columns / new_value.number_of_columns, 2
            )
            return DODatasetDriftMetric(
                drift_share=new_value.drift_share,
                number_of_columns=new_value.number_of_columns,
                number_of_drifted_columns=new_value.number_of_drifted_columns,
                share_of_drifted_columns=share_of_drifted_columns,
                dataset_drift=share_of_drifted_columns > new_value.drift_share,
            )

        accum_number_of_drifted_columns = (
            new_value.number_of_drifted_columns + accum_value.number_of_drifted_columns
        )
        accum_number_of_columns = (
            new_value.number_of_columns + accum_value.number_of_columns
        )
        share_of_drifted_columns = round(
            accum_number_of_drifted_columns / accum_number_of_columns, 2
        )

        # return {
        #     "number_of_drifted_columns": accum_number_of_drifted_columns,
        #     "number_of_columns": accum_number_of_columns,
        #     "drift_share": new_value.drift_share,
        #     "share_of_drifted_columns": share_of_drifted_columns,
        #     "dataset_drift": share_of_drifted_columns > new_value.drift_share,
        # }

        return DODatasetDriftMetric(
            drift_share=new_value.drift_share,
            number_of_columns=accum_number_of_columns,
            number_of_drifted_columns=accum_number_of_drifted_columns,
            share_of_drifted_columns=share_of_drifted_columns,
            dataset_drift=share_of_drifted_columns > new_value.drift_share,
        )
