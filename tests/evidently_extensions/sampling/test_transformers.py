import pytest

from evidently_extensions.sampling.transformers import (
    TransformerException,
    get_metric_result,
    get_test_result,
    get_test_results,
)


def test_get_metric_result():
    assert get_metric_result(
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
        },
        "DatasetDriftMetric",
    ) == {
        "drift_share": 0.5,
        "number_of_columns": 10,
        "number_of_drifted_columns": 4,
        "share_of_drifted_columns": 0.4,
        "dataset_drift": False,
    }

    with pytest.raises(TransformerException):
        get_metric_result({"random": "field"}, "metricname")


def test_get_test_result():
    assert get_test_result(
        {
            "tests": [
                {
                    "name": "Share of Drifted Columns",
                    "description": "The drift is detected for 0% features (0 out of 15). The test threshold is lt=0.3",
                    "status": "SUCCESS",
                    "group": "data_drift",
                    "parameters": {
                        "condition": {"lt": 0.3},
                        "features": {
                            "age": {
                                "stattest": "Wasserstein distance (normed)",
                                "score": 0.015,
                                "threshold": 0.1,
                                "detected": False,
                            },
                            "capital-gain": {
                                "stattest": "Wasserstein distance (normed)",
                                "score": 0.01,
                                "threshold": 0.1,
                                "detected": False,
                            },
                            "capital-loss": {
                                "stattest": "Wasserstein distance (normed)",
                                "score": 0.005,
                                "threshold": 0.1,
                                "detected": False,
                            },
                            "education-num": {
                                "stattest": "Wasserstein distance (normed)",
                                "score": 0.015,
                                "threshold": 0.1,
                                "detected": False,
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
                            "occupation": {
                                "stattest": "Jensen-Shannon distance",
                                "score": 0.013,
                                "threshold": 0.1,
                                "detected": False,
                            },
                            "race": {
                                "stattest": "Jensen-Shannon distance",
                                "score": 0.004,
                                "threshold": 0.1,
                                "detected": False,
                            },
                            "relationship": {
                                "stattest": "Jensen-Shannon distance",
                                "score": 0.013,
                                "threshold": 0.1,
                                "detected": False,
                            },
                            "sex": {
                                "stattest": "Jensen-Shannon distance",
                                "score": 0.008,
                                "threshold": 0.1,
                                "detected": False,
                            },
                            "workclass": {
                                "stattest": "Jensen-Shannon distance",
                                "score": 0.014,
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
        },
        "Share of Drifted Columns",
    ) == {
        "condition": {"lt": 0.3},
        "features": {
            "age": {
                "stattest": "Wasserstein distance (normed)",
                "score": 0.015,
                "threshold": 0.1,
                "detected": False,
            },
            "capital-gain": {
                "stattest": "Wasserstein distance (normed)",
                "score": 0.01,
                "threshold": 0.1,
                "detected": False,
            },
            "capital-loss": {
                "stattest": "Wasserstein distance (normed)",
                "score": 0.005,
                "threshold": 0.1,
                "detected": False,
            },
            "education-num": {
                "stattest": "Wasserstein distance (normed)",
                "score": 0.015,
                "threshold": 0.1,
                "detected": False,
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
            "occupation": {
                "stattest": "Jensen-Shannon distance",
                "score": 0.013,
                "threshold": 0.1,
                "detected": False,
            },
            "race": {
                "stattest": "Jensen-Shannon distance",
                "score": 0.004,
                "threshold": 0.1,
                "detected": False,
            },
            "relationship": {
                "stattest": "Jensen-Shannon distance",
                "score": 0.013,
                "threshold": 0.1,
                "detected": False,
            },
            "sex": {
                "stattest": "Jensen-Shannon distance",
                "score": 0.008,
                "threshold": 0.1,
                "detected": False,
            },
            "workclass": {
                "stattest": "Jensen-Shannon distance",
                "score": 0.014,
                "threshold": 0.1,
                "detected": False,
            },
        },
    }


def test_get_test_results():
    assert get_test_results(
        {
            "tests": [
                {
                    "name": "Drift per Column",
                    "description": "The drift score for the feature **class** is 0.001. The drift detection method is Jensen-Shannon distance. The drift detection threshold is 0.1.",
                    "status": "SUCCESS",
                    "group": "data_drift",
                    "parameters": {
                        "stattest": "Jensen-Shannon distance",
                        "score": 0.001,
                        "threshold": 0.1,
                        "detected": False,
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
                        "score": 0.017,
                        "threshold": 0.1,
                        "detected": False,
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
                        "score": 0.012,
                        "threshold": 0.1,
                        "detected": False,
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
                        "score": 0.034,
                        "threshold": 0.1,
                        "detected": False,
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
                {
                    "name": "Drift per Column",
                    "description": "The drift score for the feature **capital-gain** is 0.012. The drift detection method is Wasserstein distance (normed). The drift detection threshold is 0.1.",
                    "status": "SUCCESS",
                    "group": "data_drift",
                    "parameters": {
                        "stattest": "Wasserstein distance (normed)",
                        "score": 0.012,
                        "threshold": 0.1,
                        "detected": False,
                        "column_name": "capital-gain",
                    },
                },
                {
                    "name": "Drift per Column",
                    "description": "The drift score for the feature **capital-loss** is 0.008. The drift detection method is Wasserstein distance (normed). The drift detection threshold is 0.1.",
                    "status": "SUCCESS",
                    "group": "data_drift",
                    "parameters": {
                        "stattest": "Wasserstein distance (normed)",
                        "score": 0.008,
                        "threshold": 0.1,
                        "detected": False,
                        "column_name": "capital-loss",
                    },
                },
                {
                    "name": "Drift per Column",
                    "description": "The drift score for the feature **education-num** is 0.022. The drift detection method is Wasserstein distance (normed). The drift detection threshold is 0.1.",
                    "status": "SUCCESS",
                    "group": "data_drift",
                    "parameters": {
                        "stattest": "Wasserstein distance (normed)",
                        "score": 0.022,
                        "threshold": 0.1,
                        "detected": False,
                        "column_name": "education-num",
                    },
                },
                {
                    "name": "Drift per Column",
                    "description": "The drift score for the feature **fnlwgt** is 0.021. The drift detection method is Wasserstein distance (normed). The drift detection threshold is 0.1.",
                    "status": "SUCCESS",
                    "group": "data_drift",
                    "parameters": {
                        "stattest": "Wasserstein distance (normed)",
                        "score": 0.021,
                        "threshold": 0.1,
                        "detected": False,
                        "column_name": "fnlwgt",
                    },
                },
                {
                    "name": "Drift per Column",
                    "description": "The drift score for the feature **hours-per-week** is 0.009. The drift detection method is Wasserstein distance (normed). The drift detection threshold is 0.1.",
                    "status": "SUCCESS",
                    "group": "data_drift",
                    "parameters": {
                        "stattest": "Wasserstein distance (normed)",
                        "score": 0.009,
                        "threshold": 0.1,
                        "detected": False,
                        "column_name": "hours-per-week",
                    },
                },
            ],
            "summary": True,
            "total_tests": 16,
            "success_tests": 16,
            "failed_tests": 0,
            "by_status": {"SUCCESS": 16},
        },
        "Drift per Column",
    ) == [
        {
            "stattest": "Jensen-Shannon distance",
            "score": 0.001,
            "threshold": 0.1,
            "detected": False,
            "column_name": "class",
        },
        {
            "stattest": "Jensen-Shannon distance",
            "score": 0.017,
            "threshold": 0.1,
            "detected": False,
            "column_name": "education",
        },
        {
            "stattest": "Jensen-Shannon distance",
            "score": 0.012,
            "threshold": 0.1,
            "detected": False,
            "column_name": "marital-status",
        },
        {
            "stattest": "Jensen-Shannon distance",
            "score": 0.034,
            "threshold": 0.1,
            "detected": False,
            "column_name": "native-country",
        },
        {
            "stattest": "Jensen-Shannon distance",
            "score": 0.017,
            "threshold": 0.1,
            "detected": False,
            "column_name": "occupation",
        },
        {
            "stattest": "Jensen-Shannon distance",
            "score": 0.004,
            "threshold": 0.1,
            "detected": False,
            "column_name": "race",
        },
        {
            "stattest": "Jensen-Shannon distance",
            "score": 0.012,
            "threshold": 0.1,
            "detected": False,
            "column_name": "relationship",
        },
        {
            "stattest": "Jensen-Shannon distance",
            "score": 0.008,
            "threshold": 0.1,
            "detected": False,
            "column_name": "sex",
        },
        {
            "stattest": "Jensen-Shannon distance",
            "score": 0.017,
            "threshold": 0.1,
            "detected": False,
            "column_name": "workclass",
        },
        {
            "stattest": "Wasserstein distance (normed)",
            "score": 0.018,
            "threshold": 0.1,
            "detected": False,
            "column_name": "age",
        },
        {
            "stattest": "Wasserstein distance (normed)",
            "score": 0.012,
            "threshold": 0.1,
            "detected": False,
            "column_name": "capital-gain",
        },
        {
            "stattest": "Wasserstein distance (normed)",
            "score": 0.008,
            "threshold": 0.1,
            "detected": False,
            "column_name": "capital-loss",
        },
        {
            "stattest": "Wasserstein distance (normed)",
            "score": 0.022,
            "threshold": 0.1,
            "detected": False,
            "column_name": "education-num",
        },
        {
            "stattest": "Wasserstein distance (normed)",
            "score": 0.021,
            "threshold": 0.1,
            "detected": False,
            "column_name": "fnlwgt",
        },
        {
            "stattest": "Wasserstein distance (normed)",
            "score": 0.009,
            "threshold": 0.1,
            "detected": False,
            "column_name": "hours-per-week",
        },
    ]
