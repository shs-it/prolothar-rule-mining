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
from random import Random

from prolothar_common.models.dataset import ClassificationDataset
from prolothar_common.models.dataset.instance import ClassificationInstance
from prolothar_rule_mining.rule_miner.classification.rce import ReliableRuleMiner

class TestGreedyReliableCausalRuleMiner(unittest.TestCase):

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

        miner = ReliableRuleMiner()
        rule = miner.mine_rules(dataset)
        self.assertTrue(rule is not None)
        self.assertTrue(str(rule))

        for instance in dataset:
            predicted_class = rule.predict(instance)
            self.assertEqual(instance.get_class(), predicted_class)

    def test_mine_rules_differentiate_three_simple_classes_with_or_condition(self):
        dataset = ClassificationDataset(['category'], ['size'])
        for i in range(13):
            dataset.add_instance(ClassificationInstance(
                'As%d' % i, {'category': 'A1', 'size': 5}, 'A'))
        for i in range(13):
            dataset.add_instance(ClassificationInstance(
                'Al%d' % i, {'category': 'A2', 'size': 30}, 'A'))
        for i in range(12):
            dataset.add_instance(ClassificationInstance(
                'B%d' % i, {'category': 'B', 'size': 7}, 'B'))
        for i in range(11):
            dataset.add_instance(ClassificationInstance(
                'Csmall%d' % i, {'category': 'C', 'size': 7}, 'CS'))
        for i in range(10):
            dataset.add_instance(ClassificationInstance(
                'Clarge%d' % i, {'category': 'C', 'size': 20}, 'CL'))

        miner = ReliableRuleMiner()
        rule = miner.mine_rules(dataset)
        self.assertTrue(rule is not None)
        self.assertTrue(str(rule))

        for instance in dataset:
            predicted_class = rule.predict(instance)
            self.assertEqual(instance.get_class(), predicted_class)

    def test_mine_rules_with_less_bins_than_numerical_values(self):
        dataset = ClassificationDataset(['category'], ['size'])
        random = Random(42)
        for i in range(13):
            dataset.add_instance(ClassificationInstance(
                'A%d' % i, {'category': 'A', 'size': 5 + random.random()}, 'A'))
        for i in range(12):
            dataset.add_instance(ClassificationInstance(
                'B%d' % i, {'category': 'B', 'size': 7 + random.random()}, 'B'))
        for i in range(11):
            dataset.add_instance(ClassificationInstance(
                'Csmall%d' % i, {'category': 'C', 'size': 7 + random.random()}, 'CS'))
        for i in range(10):
            dataset.add_instance(ClassificationInstance(
                'Clarge%d' % i, {'category': 'C', 'size': 20 + random.random()}, 'CL'))

        miner = ReliableRuleMiner()
        rule = miner.mine_rules(dataset)
        self.assertTrue(rule is not None)
        self.assertTrue(str(rule))

        for instance in dataset:
            predicted_class = rule.predict(instance)
            self.assertEqual(instance.get_class(), predicted_class)

if __name__ == '__main__':
    unittest.main()