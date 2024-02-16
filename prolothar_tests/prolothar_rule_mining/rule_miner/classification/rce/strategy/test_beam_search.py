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
import unittest

from prolothar_common.models.dataset import ClassificationDataset
from prolothar_common.models.dataset.instance import ClassificationInstance
from prolothar_rule_mining.models.conditions import EqualsCondition
from prolothar_rule_mining.rule_miner.classification.rce.strategy.beam_search import BeamSearch

class TestBeamSearch(unittest.TestCase):

    def test_mine_rules_differentiate_three_simple_classes(self):
        dataset = ClassificationDataset(['category'], ['size'])
        for i in range(13):
            dataset.add_instance(ClassificationInstance(
                'A%d' % i, {'category': 'A', 'size': 5}, 'A'))
        for i in range(12):
            dataset.add_instance(ClassificationInstance(
                'B%d' % i, {'category': 'B', 'size': 7}, 'B'))
        for i in range(11):
            dataset.add_instance(ClassificationInstance(
                'Csmall%d' % i, {'category': 'C', 'size': 7}, 'CS'))
        for i in range(10):
            dataset.add_instance(ClassificationInstance(
                'Clarge%d' % i, {'category': 'C', 'size': 20}, 'CL'))

        miner = BeamSearch()
        candidate_condition = miner.create_next_condition(dataset)
        self.assertTrue(candidate_condition is not None)
        self.assertEqual(
            EqualsCondition(dataset.get_attribute_by_name('category'), 'A'),
            candidate_condition.condition
        )

if __name__ == '__main__':
    unittest.main()