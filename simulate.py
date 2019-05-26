"""
Monte Carlo random simulator.
"""

import os
import enum
import json
import logging

import typing

import begin
import colored

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class String(object):
    """
    Colored text entity.

    Usage:
    >>> print(String.info("This is an info message!"))
    >>> print(String.normal("Normal text. Lorem Ipsum Dolor Sit Amet."))
    >>> print(String.error("Something went wrong!"))
    """

    INFO = colored.fore.YELLOW_3B
    NORMAL = colored.fore.WHITE
    ERROR = colored.fore.RED

    @classmethod
    def info(cls, *strings) -> str:
        """
        Returns a colored string from parameters.
        """
        return "".join([
            colored.style.BOLD,
            cls.INFO,
            "[!]",
            " ",
            " ".join([
                str(x)
                for x in strings
            ]),
            colored.style.RESET
        ])

    @classmethod
    def normal(cls, *strings) -> str:
        """
        Returns a normal string from parameters.
        """
        return "".join([
            cls.NORMAL,
            " ".join([
                str(x)
                for x in strings
            ]),
            colored.style.RESET
        ])

    @classmethod
    def error(cls, *strings) -> str:
        """
        Returns an error string from parameters.
        """
        return "".join([
            colored.style.BOLD,
            cls.ERROR,
            "[x]",
            " ",
            " ".join([
                str(x)
                for x in strings
            ]),
            colored.style.RESET
        ])


class Distribution(object):
    """
    Distribution entity.
    """

    def __init__(self, params: dict) -> None:
        """
        Distribution construction.

        @raise: ValueError, TypeError.
        """
        if not params:
            raise ValueError("Distribution params required.")
        if not isinstance(params, dict):
            raise TypeError("Expected dict, got:", type(params))
        self._params = params

    def __str__(self) -> str:
        """
        String serializer.
        """
        return "<Distribution: {}>".format(self.__class__.__name__)

    def to_array(self, simulations: int) -> np.array:
        """
        Numpy serializer.

        @raises: ValueError, TypeError, KeyError.
        """
        if not simulations:
            raise ValueError("At least one simulation is required.")
        if not isinstance(simulations, int):
            raise TypeError("Expected int, got:", type(simulations))
        return np.array([])


class BinomialDistribution(Distribution):
    """
    Binomial Distribution entity.

    Sample configuration:
    >>> {
    ...     "Feature Name": {
    ...         "distribution": "binomial",
    ...         "success_rate": 0.02
    ...     }
    ... }
    """

    SUCCESS_RATE = "success_rate"

    @property
    def success_rate(self) -> float:
        """
        Success rate getter.

        @raises: KeyError, TypeError.
        """
        if self.SUCCESS_RATE not in self._params:
            raise KeyError("Missing 'success_rate' in binomial distribution.")
        p = self._params[self.SUCCESS_RATE]
        if not isinstance(p, float) or p < 0 or p > 1:
            raise TypeError("Expected number between 0 and 1, got:", type(p))
        return p

    def to_array(self, simulations: int) -> np.array:
        """
        Numpy serializer.

        @raises: ValueError, TypeError.
        """
        Distribution.to_array(self, simulations=simulations)
        return np.random.binomial(1, self.success_rate, simulations)


class ExponentialDistribution(Distribution):
    """
    Exponential Distribution entity.

    Sample configuration:
    >>> {
    ...     "Feature Name": {
    ...         "distribution": "exponential",
    ...         "lambda": 50
    ...     }
    ... }
    """

    LAMBDA = "lambda"

    @property
    def scale(self) -> float:
        """
        Lambda scale getter.

        @raises: KeyError, TypeError.
        """
        if self.LAMBDA not in self._params:
            raise KeyError("Missing 'lambda' in exponential distribution.")
        l = self._params[self.LAMBDA]
        if not isinstance(l, (int, float)) or l < 0:
            raise TypeError("Expected a positive number, got:", type(l))
        return l

    def to_array(self, simulations: int) -> np.array:
        """
        Numpy serializer.

        @raises: ValueError, TypeError.
        """
        Distribution.to_array(self, simulations=simulations)
        return np.random.exponential(self.scale, simulations)


class NormalDistribution(Distribution):
    """
    Normal Distribution entity.

    Sample configuration:
    >>> {
    ...     "Feature Name": {
    ...         "distribution": "normal",
    ...         "avg": 200,
    ...         "std": 0.8,
    ...         "min": 0,
    ...         "max": 3
    ...     }
    ... }
    """

    AVERAGE = "avg"
    STANDARD_DEVIATION = "std"
    MINIMUM = "min"
    MAXIMUM = "max"

    @property
    def avg(self) -> float:
        """
        Average getter.

        @raises: KeyError, TypeError.
        """
        if self.AVERAGE not in self._params:
            raise KeyError("Missing 'avg' in normal distribution.")
        avg = self._params[self.AVERAGE]
        if not isinstance(avg, (int, float)):
            raise TypeError("Expected number, got:", type(avg))
        return avg

    @property
    def std(self) -> float:
        """
        Standard deviation getter.

        @raises: KeyError, TypeError.
        """
        if self.STANDARD_DEVIATION not in self._params:
            raise KeyError("Missing 'std' in normal distribution.")
        std = self._params[self.STANDARD_DEVIATION]
        if not isinstance(std, (int, float)) or std < 0:
            raise TypeError("Expected a positive number, got:", type(std))
        return std

    @property
    def max_value(self) -> float:
        """
        Maximum value getter.

        @raises: TypeError.
        """
        if self.MAXIMUM not in self._params:
            return None
        max_value = self._params[self.MAXIMUM]
        if not isinstance(max_value, (int, float)):
            raise TypeError("Expected number, got:", type(max_value))
        return max_value

    @property
    def min_value(self) -> float:
        """
        Minimum value getter.

        @raises: TypeError.
        """
        if self.MINIMUM not in self._params:
            return None
        min_value = self._params[self.MINIMUM]
        if not isinstance(min_value, (int, float)):
            raise TypeError("Expected number, got:", type(min_value))
        return min_value

    def to_array(self, simulations: int) -> np.array:
        """
        Numpy serializer.

        @raises: ValueError, TypeError.
        """
        Distribution.to_array(self, simulations=simulations)
        sample = np.random.normal(self.avg, self.std, simulations)
        if self.max_value:
            sample = np.minimum(sample, self.max_value)
        if self.min_value:
            sample = np.maximum(sample, self.min_value)
        return sample


class LogNormalDistribution(NormalDistribution):
    """
    LogNormal Distribution entity.

    Sample configuration:
    >>> {
    ...     "Feature Name": {
    ...         "distribution": "lognormal",
    ...         "avg": 200,
    ...         "std": 0.8,
    ...         "min": 0,
    ...         "max": 3
    ...     }
    ... }
    """

    def to_array(self, simulations: int) -> np.array:
        """
        Numpy serializer.

        @raises: ValueError, TypeError.
        """
        Distribution.to_array(self, simulations=simulations)
        sample = np.random.lognormal(self.avg, self.std, simulations)
        if self.max_value:
            sample = np.minimum(sample, self.max_value)
        if self.min_value:
            sample = np.maximum(sample, self.min_value)
        return sample


class Attribute(object):
    """
    Dataset attribute entity.
    """

    def __init__(self, name: str) -> None:
        """
        Attribute constructor.

        @raises: TypeError, ValueError.
        """
        if not name:
            raise ValueError("Expected str, got:", name)
        if not isinstance(name, str):
            raise TypeError("Expected str, got:", type(name))
        if " " in name:
            raise ValueError("Space not supported in name:", name)
        self._name = name

    @property
    def title(self) -> str:
        """
        Feature name getter.
        """
        return self._name


class Target(Attribute):
    """
    Dataset target entity.
    """

    def __init__(self, name: str, formula: str) -> None:
        """
        Target constructor.

        @raises: TypeError, ValueError.
        """
        Attribute.__init__(self, name=name)
        if not formula:
            raise ValueError("Expected formula, got:", formula)
        if not isinstance(formula, str):
            raise TypeError("Expected str, got:", type(formula))
        self._formula = formula

    @property
    def formula(self):
        """
        Formula getter.
        """
        return self._formula

    def __str__(self) -> str:
        """
        String serializer.
        """
        return "<Target: '{}'>".format(self.title)


class Feature(Attribute):
    """
    Dataset feature entity.
    """

    DISTRIBUTION = "distribution"
    DISTRIBUTIONS = {
        "binomial": BinomialDistribution,
        "exponential": ExponentialDistribution,
        "normal": NormalDistribution,
        "lognormal": LogNormalDistribution,
    }

    def __init__(self, name: str, params: dict) -> None:
        """
        Feature constructor.

        @raises: TypeError, ValueError.
        """
        Attribute.__init__(self, name=name)
        if not params:
            raise ValueError("Expected dataset, got:", params)
        if not isinstance(params, dict):
            raise TypeError("Expected dict, got:", type(params))
        self._params = params

    def __str__(self) -> str:
        """
        String serializer.
        """
        return "<Feature: '{}'>".format(self.title)

    def simulate(self, simulations: int) -> np.array:
        """
        Generate a random scenario.

        @raises: ValueError, TypeError, KeyError, NotImplementedError.
        """
        if not simulations:
            raise ValueError("At least one simulation is required.")
        if not isinstance(simulations, int):
            raise TypeError("Expected int, got:", type(simulations))
        if simulations < 0:
            raise ValueError("At least one simulation is required.")
        if self.DISTRIBUTION not in self._params:
            raise KeyError("No", self.DISTRIBUTION, "in", self.title, "config.")
        if self._params[self.DISTRIBUTION] not in self.DISTRIBUTIONS:
            raise NotImplementedError("Distribution not supported:",
                                      self._params[self.DISTRIBUTION])
        d = self.DISTRIBUTIONS[self._params[self.DISTRIBUTION]](self._params)
        return d.to_array(simulations=simulations)


class Datasource(object):
    """
    Datasource entity.

    Usage:
    >>> d = Datasource.load(path="/tmp/dataset.json")
    >>> for feature in d.get_features():
    ...     print(feature)
    >>> for target in d.get_targets():
    ...     print(target)
    """

    FEATURES = "features"
    TARGETS = "targets"

    @classmethod
    def load(cls, path: str=None) -> object:
        """
        Datsource importer.

        @raises: TypeError, ValueError, OSError, NotImplementedError.
        """
        if not path:
            raise ValueError("Expected path, got:", path)
        if not isinstance(path, str):
            raise TypeError("Expected string, got:", type(path))
        if not os.path.isfile(path):
            raise OSError("File not found:", path)
        if path.endswith(".json"):
            with open(path) as file_buffer:
                return cls(json.load(file_buffer))
        raise NotImplementedError("File not supported:", path) 

    def __init__(self, data: dict) -> None:
        """
        Datasource constructor.

        @raises: TypeError, ValueError.
        """
        if not data:
            raise ValueError("Expected dataset, got:", data)
        if not isinstance(data, dict):
            raise TypeError("Expected dict, got:", type(data))
        self.__data = data

    def __str__(self) -> str:
        """
        String serializer.
        """
        return "<Datasource>"

    def get_features(self) -> typing.Generator:
        """
        Features generator.

        @raises: KeyError, TypeError.
        """
        if self.FEATURES not in self.__data:
            raise KeyError("No", self.FEATURES, "in dataset.")
        features = self.__data[self.FEATURES]
        if not isinstance(features, dict):
            raise TypeError("Expected dict, got:", type(features))
        for name, params in features.items():
            yield Feature(name=name, params=params)

    def get_targets(self) -> typing.Generator:
        """
        Targets generator.

        @raises: KeyError, TypeError.
        """
        if self.TARGETS not in self.__data:
            raise KeyError("No", self.TARGETS, "in dataset.")
        targets = self.__data[self.TARGETS]
        if not isinstance(targets, list):
            raise TypeError("Expected list, got:", type(targets))
        for target in targets:
            if not isinstance(target, dict):
                raise TypeError("Expected dict, got:", type(target))
            for name, formula in target.items():
                yield Target(name=name, formula=formula)


@begin.start(lexical_order=True, short_args=True)
@begin.logging
def run(path: "Dataset file path."="dataset.json",
        simulations: "Amount of simulations to run."=10):
    """
    Main task.
    """
    print(String.info("Loading dataset from:", path))
    simulations = int(simulations)
    dataset = pd.DataFrame()
    datasource = Datasource.load(path=path)
    for feature in datasource.get_features():
        print(String.normal("Simulating", feature.title))
        dataset[feature.title] = feature.simulate(simulations=simulations)
    for target in datasource.get_targets():
        print(String.normal("Calculating", target.title))
        dataset[target.title] = dataset.eval(target.formula)
    print(String.info("All scenarios generated."))
    print(String.normal(dataset.describe()))
    # print(String.normal(dataset))
