'''
    This file is part of Prolothar-Rule-Mining (More Info: https://github.com/shs-it/prolothar-rule-mining).

    Prolothar-Rule-Mining is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Prolothar-Rule-Mining is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Prolothar-Rule-Mining. If not, see <https://www.gnu.org/licenses/>.
'''
from typing import Set

from prolothar_common.models.dataset.instance import Instance

from prolothar_rule_mining.rule_miner.classification.rules.rule import Rule

class VoteForClassRule(Rule):
    """
    wrapper for rules that vote for predicting a class - the majority wins
    """

    def __init__(self, class_label):
        self.__class_label = class_label

    def predict(self, instance: Instance) -> str:
        return self.__class_label

    def __eq__(self, other) -> bool:
        try:
            return self.__class_label == other.__class_label
        except AttributeError:
            return False

    def __hash__(self) -> int:
        return hash(self.__class_label)

    def to_string(self, prefix='') -> str:
        return prefix + str(self.__class_label)

    def to_html(self) -> str:
        return '<div id="%d" class="Rule VoteForClassRule">%s</div>' % (
            id(self), self.__class_label)

    def count_nr_of_terms(self) -> int:
        return 0

    def get_set_of_output_classes(self) -> Set[str]:
        return set([self.__class_label])
