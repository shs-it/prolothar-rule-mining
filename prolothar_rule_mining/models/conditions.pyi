from typing import Callable, Any
from prolothar_common.models.dataset.instance import Instance
from prolothar_common.models.dataset import Dataset
from prolothar_common.func_tools import identity

class Condition:

    def check_instance(self, instance: Instance) -> bool: 
        """returns True iff the given instance fullfils this condition"""
        ...

    def divide_dataset(self, dataset: Dataset) -> tuple[Dataset, Dataset]:
        """splits a dataset into matching and non-matching instances"""
        ...

    def to_html(self) -> str:
        """
        returns a human readable html string representation of this condition
        """
        ...

    def to_bal(self, attribute_formatter: Callable[[str], str] = identity,
               operator_formatter: Callable[[str], str] = identity,
               join_operator_formatter: Callable[[str], str] = identity,
               numerical_value_formatter: Callable[[float], str] = str,
               categorical_value_formatter: Callable[[Any], str] = str) -> str:
        """
        export to the BAL language of IBM ODM
        """
        ...       

    