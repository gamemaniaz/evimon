from __future__ import annotations

import json
from abc import ABC, abstractclassmethod
from dataclasses import dataclass


class DOBase(ABC):
    @abstractclassmethod
    def from_dict(cls, input_dict: dict) -> DOBase:
        return


class DOSerializerException(Exception):
    """Generic wrapper exception class for serialisation"""


@dataclass(frozen=True)
class DODatasetDriftMetric(DOBase):
    drift_share: float
    number_of_columns: int
    number_of_drifted_columns: int
    share_of_drifted_columns: float
    dataset_drift: bool

    @classmethod
    def from_dict(cls, input_dict: dict) -> DODatasetDriftMetric:
        try:
            metric = next(
                iter(
                    [
                        x
                        for x in input_dict["metrics"]
                        if x["metric"] == "DatasetDriftMetric"
                    ]
                )
            )["result"]
            return DODatasetDriftMetric(
                drift_share=metric["drift_share"],
                number_of_columns=metric["number_of_columns"],
                number_of_drifted_columns=metric["number_of_drifted_columns"],
                share_of_drifted_columns=metric["share_of_drifted_columns"],
                dataset_drift=metric["dataset_drift"],
            )
        except:
            raise DOSerializerException(
                f"unexpected dict schema found : {json.dumps(input_dict)}"
            )
