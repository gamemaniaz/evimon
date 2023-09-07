from functools import wraps


class TransformerException(Exception):
    """General exception wrapper for transformations"""


def wrap_exception(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as err:
            raise TransformerException() from err

    return wrapper


@wrap_exception
def get_metric_result(input: dict, metric_name: str) -> dict:
    return [x for x in input["metrics"] if x["metric"] == metric_name][0]["result"]


@wrap_exception
def get_test_result(input: dict, test_name: str) -> dict:
    return [x for x in input["tests"] if x["name"] == test_name][0]["parameters"]


@wrap_exception
def get_test_results(input: dict, test_name: str) -> list[dict]:
    return [x["parameters"] for x in input["tests"] if x["name"] == test_name]
