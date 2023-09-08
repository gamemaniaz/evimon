from typing import Any

import pytest

from evidently_extensions.sampling.accumulators import (
    AggDriftPerColumnSampleAccumulator,
    DataDriftTableSampleAccumulator,
    DatasetDriftMetricSampleAccumulator,
    NumberOfDriftedFeaturesSampleAccumulator,
    SampleAccumulator,
    ShareOfDriftedColumnsSampleAccumulator,
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


def test_data_drift_table_sample_accumulator():
    accumulator = DataDriftTableSampleAccumulator(drift_share=0.5)

    assert accumulator.accum_value == {
        "drift_share": 0.5,
        "number_of_columns": 0,
        "number_of_drifted_columns": 0,
        "share_of_drifted_columns": 0.0,
        "dataset_drift": False,
    }
    assert accumulator.accumulate(
        {
            "metrics": [
                {
                    "metric": "DataDriftTable",
                    "result": {
                        "number_of_columns": 3,
                        "number_of_drifted_columns": 1,
                        "share_of_drifted_columns": 0.33,
                        "dataset_drift": False,
                        "drift_by_columns": {
                            "age": {
                                "column_name": "age",
                                "column_type": "num",
                                "stattest_name": "Wasserstein distance (normed)",
                                "stattest_threshold": 0.1,
                                "drift_score": 0.2,
                                "drift_detected": True,
                                "current": {
                                    "small_distribution": {
                                        "x": [
                                            17.0,
                                            24.3,
                                            31.6,
                                            38.9,
                                            46.2,
                                            53.5,
                                            60.8,
                                            68.1,
                                            75.4,
                                            82.7,
                                            90.0,
                                        ],
                                        "y": [
                                            0.023779958084449524,
                                            0.024365429557264368,
                                            0.025732698266233173,
                                            0.025413668900807096,
                                            0.016820910937518634,
                                            0.010780387897638556,
                                            0.006931000609311032,
                                            0.00212452522470535,
                                            0.0007677739673440142,
                                            0.0002699479245912744,
                                        ],
                                    }
                                },
                                "reference": {
                                    "small_distribution": {
                                        "x": [
                                            17.0,
                                            24.3,
                                            31.6,
                                            38.9,
                                            46.2,
                                            53.5,
                                            60.8,
                                            68.1,
                                            75.4,
                                            82.7,
                                            90.0,
                                        ],
                                        "y": [
                                            0.023125553947471754,
                                            0.024345640783996945,
                                            0.02496269619557292,
                                            0.026757766483793865,
                                            0.01636599239338966,
                                            0.011233213288007813,
                                            0.006717489594201926,
                                            0.0023981017131702027,
                                            0.0008133912243501288,
                                            0.0002664557459078008,
                                        ],
                                    }
                                },
                            },
                            "capital-gain": {
                                "column_name": "capital-gain",
                                "column_type": "num",
                                "stattest_name": "Wasserstein distance (normed)",
                                "stattest_threshold": 0.1,
                                "drift_score": 0.009828174689857543,
                                "drift_detected": False,
                                "current": {
                                    "small_distribution": {
                                        "x": [
                                            0.0,
                                            9999.9,
                                            19999.8,
                                            29999.699999999997,
                                            39999.6,
                                            49999.5,
                                            59999.399999999994,
                                            69999.3,
                                            79999.2,
                                            89999.09999999999,
                                            99999.0,
                                        ],
                                        "y": [
                                            9.767206235301336e-05,
                                            1.5458003789230707e-06,
                                            2.5848648720402345e-07,
                                            1.023708860213954e-08,
                                            2.559272150534885e-09,
                                            0.0,
                                            0.0,
                                            0.0,
                                            0.0,
                                            5.118544301069766e-07,
                                        ],
                                    }
                                },
                                "reference": {
                                    "small_distribution": {
                                        "x": [
                                            0.0,
                                            9999.9,
                                            19999.8,
                                            29999.699999999997,
                                            39999.6,
                                            49999.5,
                                            59999.399999999994,
                                            69999.3,
                                            79999.2,
                                            89999.09999999999,
                                            99999.0,
                                        ],
                                        "y": [
                                            9.770777478454555e-05,
                                            1.52540427943182e-06,
                                            2.764155405681822e-07,
                                            2.0475225227272746e-08,
                                            2.0475225227272746e-08,
                                            0.0,
                                            0.0,
                                            0.0,
                                            0.0,
                                            4.504549550000001e-07,
                                        ],
                                    }
                                },
                            },
                            "capital-loss": {
                                "column_name": "capital-loss",
                                "column_type": "num",
                                "stattest_name": "Wasserstein distance (normed)",
                                "stattest_threshold": 0.1,
                                "drift_score": 0.005491838706105067,
                                "drift_detected": False,
                                "current": {
                                    "small_distribution": {
                                        "x": [
                                            0.0,
                                            435.6,
                                            871.2,
                                            1306.8000000000002,
                                            1742.4,
                                            2178.0,
                                            2613.6000000000004,
                                            3049.2000000000003,
                                            3484.8,
                                            3920.4,
                                            4356.0,
                                        ],
                                        "y": [
                                            0.0021885788228904086,
                                            1.2337965499100315e-06,
                                            1.351300983234796e-06,
                                            3.343001128089562e-05,
                                            5.416954376271663e-05,
                                            1.5275576332219428e-05,
                                            1.0575398999228844e-06,
                                            5.875221666238246e-08,
                                            3.525132999742948e-07,
                                            1.762566499871474e-07,
                                        ],
                                    }
                                },
                                "reference": {
                                    "small_distribution": {
                                        "x": [
                                            0.0,
                                            368.3,
                                            736.6,
                                            1104.9,
                                            1473.2,
                                            1841.5,
                                            2209.8,
                                            2578.1,
                                            2946.4,
                                            3314.7000000000003,
                                            3683.0,
                                        ],
                                        "y": [
                                            0.0025909267695854716,
                                            1.1118664390453692e-06,
                                            1.111866439045369e-06,
                                            3.057632707374766e-06,
                                            3.724752570801987e-05,
                                            6.810181939152883e-05,
                                            1.1674597609976386e-05,
                                            1.1118664390453688e-06,
                                            2.779666097613422e-07,
                                            5.55933219522685e-07,
                                        ],
                                    }
                                },
                            },
                        },
                    },
                }
            ]
        }
    ) == {
        "drift_share": 0.5,
        "number_of_columns": 3,
        "number_of_drifted_columns": 1,
        "share_of_drifted_columns": 0.33,
        "dataset_drift": False,
    }
    assert accumulator.accumulate(
        {
            "metrics": [
                {
                    "metric": "DataDriftTable",
                    "result": {
                        "number_of_columns": 3,
                        "number_of_drifted_columns": 3,
                        "share_of_drifted_columns": 1,
                        "dataset_drift": False,
                        "drift_by_columns": {
                            "age": {
                                "column_name": "age",
                                "column_type": "num",
                                "stattest_name": "Wasserstein distance (normed)",
                                "stattest_threshold": 0.1,
                                "drift_score": 0.2,
                                "drift_detected": True,
                                "current": {
                                    "small_distribution": {
                                        "x": [
                                            17.0,
                                            24.3,
                                            31.6,
                                            38.9,
                                            46.2,
                                            53.5,
                                            60.8,
                                            68.1,
                                            75.4,
                                            82.7,
                                            90.0,
                                        ],
                                        "y": [
                                            0.023779958084449524,
                                            0.024365429557264368,
                                            0.025732698266233173,
                                            0.025413668900807096,
                                            0.016820910937518634,
                                            0.010780387897638556,
                                            0.006931000609311032,
                                            0.00212452522470535,
                                            0.0007677739673440142,
                                            0.0002699479245912744,
                                        ],
                                    }
                                },
                                "reference": {
                                    "small_distribution": {
                                        "x": [
                                            17.0,
                                            24.3,
                                            31.6,
                                            38.9,
                                            46.2,
                                            53.5,
                                            60.8,
                                            68.1,
                                            75.4,
                                            82.7,
                                            90.0,
                                        ],
                                        "y": [
                                            0.023125553947471754,
                                            0.024345640783996945,
                                            0.02496269619557292,
                                            0.026757766483793865,
                                            0.01636599239338966,
                                            0.011233213288007813,
                                            0.006717489594201926,
                                            0.0023981017131702027,
                                            0.0008133912243501288,
                                            0.0002664557459078008,
                                        ],
                                    }
                                },
                            },
                            "capital-gain": {
                                "column_name": "capital-gain",
                                "column_type": "num",
                                "stattest_name": "Wasserstein distance (normed)",
                                "stattest_threshold": 0.1,
                                "drift_score": 0.2,
                                "drift_detected": True,
                                "current": {
                                    "small_distribution": {
                                        "x": [
                                            0.0,
                                            9999.9,
                                            19999.8,
                                            29999.699999999997,
                                            39999.6,
                                            49999.5,
                                            59999.399999999994,
                                            69999.3,
                                            79999.2,
                                            89999.09999999999,
                                            99999.0,
                                        ],
                                        "y": [
                                            9.767206235301336e-05,
                                            1.5458003789230707e-06,
                                            2.5848648720402345e-07,
                                            1.023708860213954e-08,
                                            2.559272150534885e-09,
                                            0.0,
                                            0.0,
                                            0.0,
                                            0.0,
                                            5.118544301069766e-07,
                                        ],
                                    }
                                },
                                "reference": {
                                    "small_distribution": {
                                        "x": [
                                            0.0,
                                            9999.9,
                                            19999.8,
                                            29999.699999999997,
                                            39999.6,
                                            49999.5,
                                            59999.399999999994,
                                            69999.3,
                                            79999.2,
                                            89999.09999999999,
                                            99999.0,
                                        ],
                                        "y": [
                                            9.770777478454555e-05,
                                            1.52540427943182e-06,
                                            2.764155405681822e-07,
                                            2.0475225227272746e-08,
                                            2.0475225227272746e-08,
                                            0.0,
                                            0.0,
                                            0.0,
                                            0.0,
                                            4.504549550000001e-07,
                                        ],
                                    }
                                },
                            },
                            "capital-loss": {
                                "column_name": "capital-loss",
                                "column_type": "num",
                                "stattest_name": "Wasserstein distance (normed)",
                                "stattest_threshold": 0.1,
                                "drift_score": 0.2,
                                "drift_detected": True,
                                "current": {
                                    "small_distribution": {
                                        "x": [
                                            0.0,
                                            435.6,
                                            871.2,
                                            1306.8000000000002,
                                            1742.4,
                                            2178.0,
                                            2613.6000000000004,
                                            3049.2000000000003,
                                            3484.8,
                                            3920.4,
                                            4356.0,
                                        ],
                                        "y": [
                                            0.0021885788228904086,
                                            1.2337965499100315e-06,
                                            1.351300983234796e-06,
                                            3.343001128089562e-05,
                                            5.416954376271663e-05,
                                            1.5275576332219428e-05,
                                            1.0575398999228844e-06,
                                            5.875221666238246e-08,
                                            3.525132999742948e-07,
                                            1.762566499871474e-07,
                                        ],
                                    }
                                },
                                "reference": {
                                    "small_distribution": {
                                        "x": [
                                            0.0,
                                            368.3,
                                            736.6,
                                            1104.9,
                                            1473.2,
                                            1841.5,
                                            2209.8,
                                            2578.1,
                                            2946.4,
                                            3314.7000000000003,
                                            3683.0,
                                        ],
                                        "y": [
                                            0.0025909267695854716,
                                            1.1118664390453692e-06,
                                            1.111866439045369e-06,
                                            3.057632707374766e-06,
                                            3.724752570801987e-05,
                                            6.810181939152883e-05,
                                            1.1674597609976386e-05,
                                            1.1118664390453688e-06,
                                            2.779666097613422e-07,
                                            5.55933219522685e-07,
                                        ],
                                    }
                                },
                            },
                        },
                    },
                }
            ]
        }
    ) == {
        "drift_share": 0.5,
        "number_of_columns": 6,
        "number_of_drifted_columns": 4,
        "share_of_drifted_columns": 0.67,
        "dataset_drift": True,
    }

    with pytest.raises(TransformerException):
        accumulator.accumulate({"some_unknown_field": 20})


def test_share_of_drifted_columns_sample_accumulator():
    accumulator = ShareOfDriftedColumnsSampleAccumulator()

    assert accumulator.accum_value == {
        "drift_share": 0.0,
        "number_of_columns": 0,
        "number_of_drifted_columns": 0,
        "share_of_drifted_columns": 0.0,
        "dataset_drift": False,
    }
    assert accumulator.accumulate(
        {
            "tests": [
                {
                    "name": "Share of Drifted Columns",
                    "description": "The drift is detected for 0% features (0 out of 15). The test threshold is lt=0.3",
                    "status": "SUCCESS",
                    "group": "data_drift",
                    "parameters": {
                        "condition": {"lt": 0.5},
                        "features": {
                            "age": {
                                "stattest": "Wasserstein distance (normed)",
                                "score": 0.2,
                                "threshold": 0.1,
                                "detected": True,
                            },
                            "capital-gain": {
                                "stattest": "Wasserstein distance (normed)",
                                "score": 0.2,
                                "threshold": 0.1,
                                "detected": True,
                            },
                            "capital-loss": {
                                "stattest": "Wasserstein distance (normed)",
                                "score": 0.2,
                                "threshold": 0.1,
                                "detected": True,
                            },
                            "education-num": {
                                "stattest": "Wasserstein distance (normed)",
                                "score": 0.2,
                                "threshold": 0.1,
                                "detected": True,
                            },
                            "fnlwgt": {
                                "stattest": "Wasserstein distance (normed)",
                                "score": 0.018,
                                "threshold": 0.1,
                                "detected": False,
                            },
                            "hours-per-week": {
                                "stattest": "Wasserstein distance (normed)",
                                "score": 0.008,
                                "threshold": 0.1,
                                "detected": False,
                            },
                            "class": {
                                "stattest": "Jensen-Shannon distance",
                                "score": 0.001,
                                "threshold": 0.1,
                                "detected": False,
                            },
                            "education": {
                                "stattest": "Jensen-Shannon distance",
                                "score": 0.014,
                                "threshold": 0.1,
                                "detected": False,
                            },
                            "marital-status": {
                                "stattest": "Jensen-Shannon distance",
                                "score": 0.013,
                                "threshold": 0.1,
                                "detected": False,
                            },
                            "native-country": {
                                "stattest": "Jensen-Shannon distance",
                                "score": 0.024,
                                "threshold": 0.1,
                                "detected": False,
                            },
                        },
                    },
                }
            ],
            "summary": {
                "all_passed": True,
                "total_tests": 1,
                "success_tests": 1,
                "failed_tests": 0,
                "by_status": {"SUCCESS": 1},
            },
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
            "tests": [
                {
                    "name": "Share of Drifted Columns",
                    "description": "The drift is detected for 0% features (0 out of 15). The test threshold is lt=0.3",
                    "status": "SUCCESS",
                    "group": "data_drift",
                    "parameters": {
                        "condition": {"lt": 0.5},
                        "features": {
                            "age": {
                                "stattest": "Wasserstein distance (normed)",
                                "score": 0.2,
                                "threshold": 0.1,
                                "detected": True,
                            },
                            "capital-gain": {
                                "stattest": "Wasserstein distance (normed)",
                                "score": 0.2,
                                "threshold": 0.1,
                                "detected": True,
                            },
                            "capital-loss": {
                                "stattest": "Wasserstein distance (normed)",
                                "score": 0.2,
                                "threshold": 0.1,
                                "detected": True,
                            },
                            "education-num": {
                                "stattest": "Wasserstein distance (normed)",
                                "score": 0.2,
                                "threshold": 0.1,
                                "detected": True,
                            },
                            "fnlwgt": {
                                "stattest": "Wasserstein distance (normed)",
                                "score": 0.2,
                                "threshold": 0.1,
                                "detected": True,
                            },
                            "hours-per-week": {
                                "stattest": "Wasserstein distance (normed)",
                                "score": 0.2,
                                "threshold": 0.1,
                                "detected": True,
                            },
                            "class": {
                                "stattest": "Jensen-Shannon distance",
                                "score": 0.001,
                                "threshold": 0.1,
                                "detected": False,
                            },
                            "education": {
                                "stattest": "Jensen-Shannon distance",
                                "score": 0.014,
                                "threshold": 0.1,
                                "detected": False,
                            },
                            "marital-status": {
                                "stattest": "Jensen-Shannon distance",
                                "score": 0.013,
                                "threshold": 0.1,
                                "detected": False,
                            },
                            "native-country": {
                                "stattest": "Jensen-Shannon distance",
                                "score": 0.024,
                                "threshold": 0.1,
                                "detected": False,
                            },
                        },
                    },
                }
            ],
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


def test_number_of_drifted_features_sample_accumulator():
    accumulator = NumberOfDriftedFeaturesSampleAccumulator()

    assert accumulator.accum_value == {
        "num_cols_drift_share": 0.0,
        "number_of_columns": 0,
        "number_of_drifted_columns": 0,
        "dataset_drift": False,
    }
    assert accumulator.accumulate(
        {
            "tests": [
                {
                    "name": "Number of Drifted Features",
                    "description": "The drift is detected for 0 out of 15 features. The test threshold is lt=5.",
                    "status": "SUCCESS",
                    "group": "data_drift",
                    "parameters": {
                        "condition": {"lt": 5},
                        "features": {
                            "age": {
                                "stattest": "Wasserstein distance (normed)",
                                "score": 0.2,
                                "threshold": 0.1,
                                "detected": True,
                            },
                            "capital-gain": {
                                "stattest": "Wasserstein distance (normed)",
                                "score": 0.2,
                                "threshold": 0.1,
                                "detected": True,
                            },
                            "capital-loss": {
                                "stattest": "Wasserstein distance (normed)",
                                "score": 0.2,
                                "threshold": 0.1,
                                "detected": True,
                            },
                            "education-num": {
                                "stattest": "Wasserstein distance (normed)",
                                "score": 0.2,
                                "threshold": 0.1,
                                "detected": True,
                            },
                            "fnlwgt": {
                                "stattest": "Wasserstein distance (normed)",
                                "score": 0.018,
                                "threshold": 0.1,
                                "detected": False,
                            },
                            "hours-per-week": {
                                "stattest": "Wasserstein distance (normed)",
                                "score": 0.008,
                                "threshold": 0.1,
                                "detected": False,
                            },
                            "class": {
                                "stattest": "Jensen-Shannon distance",
                                "score": 0.001,
                                "threshold": 0.1,
                                "detected": False,
                            },
                            "education": {
                                "stattest": "Jensen-Shannon distance",
                                "score": 0.014,
                                "threshold": 0.1,
                                "detected": False,
                            },
                            "marital-status": {
                                "stattest": "Jensen-Shannon distance",
                                "score": 0.013,
                                "threshold": 0.1,
                                "detected": False,
                            },
                            "native-country": {
                                "stattest": "Jensen-Shannon distance",
                                "score": 0.024,
                                "threshold": 0.1,
                                "detected": False,
                            },
                        },
                    },
                }
            ],
        }
    ) == {
        "num_cols_drift_share": 5,
        "number_of_columns": 10,
        "number_of_drifted_columns": 4,
        "dataset_drift": False,
    }
    assert accumulator.accumulate(
        {
            "tests": [
                {
                    "name": "Number of Drifted Features",
                    "description": "The drift is detected for 0 out of 15 features. The test threshold is lt=5.",
                    "status": "SUCCESS",
                    "group": "data_drift",
                    "parameters": {
                        "condition": {"lt": 5},
                        "features": {
                            "age": {
                                "stattest": "Wasserstein distance (normed)",
                                "score": 0.2,
                                "threshold": 0.1,
                                "detected": True,
                            },
                            "capital-gain": {
                                "stattest": "Wasserstein distance (normed)",
                                "score": 0.2,
                                "threshold": 0.1,
                                "detected": True,
                            },
                            "capital-loss": {
                                "stattest": "Wasserstein distance (normed)",
                                "score": 0.2,
                                "threshold": 0.1,
                                "detected": True,
                            },
                            "education-num": {
                                "stattest": "Wasserstein distance (normed)",
                                "score": 0.2,
                                "threshold": 0.1,
                                "detected": True,
                            },
                            "fnlwgt": {
                                "stattest": "Wasserstein distance (normed)",
                                "score": 0.2,
                                "threshold": 0.1,
                                "detected": True,
                            },
                            "hours-per-week": {
                                "stattest": "Wasserstein distance (normed)",
                                "score": 0.2,
                                "threshold": 0.1,
                                "detected": True,
                            },
                            "class": {
                                "stattest": "Jensen-Shannon distance",
                                "score": 0.001,
                                "threshold": 0.1,
                                "detected": False,
                            },
                            "education": {
                                "stattest": "Jensen-Shannon distance",
                                "score": 0.014,
                                "threshold": 0.1,
                                "detected": False,
                            },
                            "marital-status": {
                                "stattest": "Jensen-Shannon distance",
                                "score": 0.013,
                                "threshold": 0.1,
                                "detected": False,
                            },
                            "native-country": {
                                "stattest": "Jensen-Shannon distance",
                                "score": 0.024,
                                "threshold": 0.1,
                                "detected": False,
                            },
                        },
                    },
                }
            ],
        }
    ) == {
        "num_cols_drift_share": 10,
        "number_of_columns": 20,
        "number_of_drifted_columns": 10,
        "dataset_drift": True,
    }

    with pytest.raises(TransformerException):
        accumulator.accumulate({"some_unknown_field": 20})


def test_agg_drift_per_column_sample_accumulator():
    accumulator = AggDriftPerColumnSampleAccumulator(drift_share=0.5)

    assert accumulator.accum_value == {
        "drift_share": 0.5,
        "number_of_columns": 0,
        "number_of_drifted_columns": 0,
        "share_of_drifted_columns": 0.0,
        "dataset_drift": False,
    }
    assert accumulator.accumulate(
        {
            "tests": [
                {
                    "name": "Drift per Column",
                    "description": "The drift score for the feature **class** is 0.001. The drift detection method is Jensen-Shannon distance. The drift detection threshold is 0.1.",
                    "status": "SUCCESS",
                    "group": "data_drift",
                    "parameters": {
                        "stattest": "Jensen-Shannon distance",
                        "score": 0.2,
                        "threshold": 0.1,
                        "detected": True,
                        "column_name": "class",
                    },
                },
                {
                    "name": "Drift per Column",
                    "description": "The drift score for the feature **education** is 0.017. The drift detection method is Jensen-Shannon distance. The drift detection threshold is 0.1.",
                    "status": "SUCCESS",
                    "group": "data_drift",
                    "parameters": {
                        "stattest": "Jensen-Shannon distance",
                        "score": 0.2,
                        "threshold": 0.1,
                        "detected": True,
                        "column_name": "education",
                    },
                },
                {
                    "name": "Drift per Column",
                    "description": "The drift score for the feature **marital-status** is 0.012. The drift detection method is Jensen-Shannon distance. The drift detection threshold is 0.1.",
                    "status": "SUCCESS",
                    "group": "data_drift",
                    "parameters": {
                        "stattest": "Jensen-Shannon distance",
                        "score": 0.2,
                        "threshold": 0.1,
                        "detected": True,
                        "column_name": "marital-status",
                    },
                },
                {
                    "name": "Drift per Column",
                    "description": "The drift score for the feature **native-country** is 0.034. The drift detection method is Jensen-Shannon distance. The drift detection threshold is 0.1.",
                    "status": "SUCCESS",
                    "group": "data_drift",
                    "parameters": {
                        "stattest": "Jensen-Shannon distance",
                        "score": 0.2,
                        "threshold": 0.1,
                        "detected": True,
                        "column_name": "native-country",
                    },
                },
                {
                    "name": "Drift per Column",
                    "description": "The drift score for the feature **occupation** is 0.017. The drift detection method is Jensen-Shannon distance. The drift detection threshold is 0.1.",
                    "status": "SUCCESS",
                    "group": "data_drift",
                    "parameters": {
                        "stattest": "Jensen-Shannon distance",
                        "score": 0.017,
                        "threshold": 0.1,
                        "detected": False,
                        "column_name": "occupation",
                    },
                },
                {
                    "name": "Drift per Column",
                    "description": "The drift score for the feature **race** is 0.004. The drift detection method is Jensen-Shannon distance. The drift detection threshold is 0.1.",
                    "status": "SUCCESS",
                    "group": "data_drift",
                    "parameters": {
                        "stattest": "Jensen-Shannon distance",
                        "score": 0.004,
                        "threshold": 0.1,
                        "detected": False,
                        "column_name": "race",
                    },
                },
                {
                    "name": "Drift per Column",
                    "description": "The drift score for the feature **relationship** is 0.012. The drift detection method is Jensen-Shannon distance. The drift detection threshold is 0.1.",
                    "status": "SUCCESS",
                    "group": "data_drift",
                    "parameters": {
                        "stattest": "Jensen-Shannon distance",
                        "score": 0.012,
                        "threshold": 0.1,
                        "detected": False,
                        "column_name": "relationship",
                    },
                },
                {
                    "name": "Drift per Column",
                    "description": "The drift score for the feature **sex** is 0.008. The drift detection method is Jensen-Shannon distance. The drift detection threshold is 0.1.",
                    "status": "SUCCESS",
                    "group": "data_drift",
                    "parameters": {
                        "stattest": "Jensen-Shannon distance",
                        "score": 0.008,
                        "threshold": 0.1,
                        "detected": False,
                        "column_name": "sex",
                    },
                },
                {
                    "name": "Drift per Column",
                    "description": "The drift score for the feature **workclass** is 0.017. The drift detection method is Jensen-Shannon distance. The drift detection threshold is 0.1.",
                    "status": "SUCCESS",
                    "group": "data_drift",
                    "parameters": {
                        "stattest": "Jensen-Shannon distance",
                        "score": 0.017,
                        "threshold": 0.1,
                        "detected": False,
                        "column_name": "workclass",
                    },
                },
                {
                    "name": "Drift per Column",
                    "description": "The drift score for the feature **age** is 0.018. The drift detection method is Wasserstein distance (normed). The drift detection threshold is 0.1.",
                    "status": "SUCCESS",
                    "group": "data_drift",
                    "parameters": {
                        "stattest": "Wasserstein distance (normed)",
                        "score": 0.018,
                        "threshold": 0.1,
                        "detected": False,
                        "column_name": "age",
                    },
                },
            ],
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
            "tests": [
                {
                    "name": "Drift per Column",
                    "description": "The drift score for the feature **class** is 0.001. The drift detection method is Jensen-Shannon distance. The drift detection threshold is 0.1.",
                    "status": "SUCCESS",
                    "group": "data_drift",
                    "parameters": {
                        "stattest": "Jensen-Shannon distance",
                        "score": 0.2,
                        "threshold": 0.1,
                        "detected": True,
                        "column_name": "class",
                    },
                },
                {
                    "name": "Drift per Column",
                    "description": "The drift score for the feature **education** is 0.017. The drift detection method is Jensen-Shannon distance. The drift detection threshold is 0.1.",
                    "status": "SUCCESS",
                    "group": "data_drift",
                    "parameters": {
                        "stattest": "Jensen-Shannon distance",
                        "score": 0.2,
                        "threshold": 0.1,
                        "detected": True,
                        "column_name": "education",
                    },
                },
                {
                    "name": "Drift per Column",
                    "description": "The drift score for the feature **marital-status** is 0.012. The drift detection method is Jensen-Shannon distance. The drift detection threshold is 0.1.",
                    "status": "SUCCESS",
                    "group": "data_drift",
                    "parameters": {
                        "stattest": "Jensen-Shannon distance",
                        "score": 0.2,
                        "threshold": 0.1,
                        "detected": True,
                        "column_name": "marital-status",
                    },
                },
                {
                    "name": "Drift per Column",
                    "description": "The drift score for the feature **native-country** is 0.034. The drift detection method is Jensen-Shannon distance. The drift detection threshold is 0.1.",
                    "status": "SUCCESS",
                    "group": "data_drift",
                    "parameters": {
                        "stattest": "Jensen-Shannon distance",
                        "score": 0.2,
                        "threshold": 0.1,
                        "detected": True,
                        "column_name": "native-country",
                    },
                },
                {
                    "name": "Drift per Column",
                    "description": "The drift score for the feature **occupation** is 0.017. The drift detection method is Jensen-Shannon distance. The drift detection threshold is 0.1.",
                    "status": "SUCCESS",
                    "group": "data_drift",
                    "parameters": {
                        "stattest": "Jensen-Shannon distance",
                        "score": 0.2,
                        "threshold": 0.1,
                        "detected": True,
                        "column_name": "occupation",
                    },
                },
                {
                    "name": "Drift per Column",
                    "description": "The drift score for the feature **race** is 0.004. The drift detection method is Jensen-Shannon distance. The drift detection threshold is 0.1.",
                    "status": "SUCCESS",
                    "group": "data_drift",
                    "parameters": {
                        "stattest": "Jensen-Shannon distance",
                        "score": 0.2,
                        "threshold": 0.1,
                        "detected": True,
                        "column_name": "race",
                    },
                },
                {
                    "name": "Drift per Column",
                    "description": "The drift score for the feature **relationship** is 0.012. The drift detection method is Jensen-Shannon distance. The drift detection threshold is 0.1.",
                    "status": "SUCCESS",
                    "group": "data_drift",
                    "parameters": {
                        "stattest": "Jensen-Shannon distance",
                        "score": 0.012,
                        "threshold": 0.1,
                        "detected": False,
                        "column_name": "relationship",
                    },
                },
                {
                    "name": "Drift per Column",
                    "description": "The drift score for the feature **sex** is 0.008. The drift detection method is Jensen-Shannon distance. The drift detection threshold is 0.1.",
                    "status": "SUCCESS",
                    "group": "data_drift",
                    "parameters": {
                        "stattest": "Jensen-Shannon distance",
                        "score": 0.008,
                        "threshold": 0.1,
                        "detected": False,
                        "column_name": "sex",
                    },
                },
                {
                    "name": "Drift per Column",
                    "description": "The drift score for the feature **workclass** is 0.017. The drift detection method is Jensen-Shannon distance. The drift detection threshold is 0.1.",
                    "status": "SUCCESS",
                    "group": "data_drift",
                    "parameters": {
                        "stattest": "Jensen-Shannon distance",
                        "score": 0.017,
                        "threshold": 0.1,
                        "detected": False,
                        "column_name": "workclass",
                    },
                },
                {
                    "name": "Drift per Column",
                    "description": "The drift score for the feature **age** is 0.018. The drift detection method is Wasserstein distance (normed). The drift detection threshold is 0.1.",
                    "status": "SUCCESS",
                    "group": "data_drift",
                    "parameters": {
                        "stattest": "Wasserstein distance (normed)",
                        "score": 0.018,
                        "threshold": 0.1,
                        "detected": False,
                        "column_name": "age",
                    },
                },
            ],
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
