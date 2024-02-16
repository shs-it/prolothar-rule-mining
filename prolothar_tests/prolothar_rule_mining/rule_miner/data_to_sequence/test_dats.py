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

from prolothar_common.models.dataset import TargetSequenceDataset
from prolothar_common.models.dataset.instance import TargetSequenceInstance

from prolothar_rule_mining.models.dataset_generator import TargetSequenceDatasetGenerator as DatasetGenerator
from prolothar_rule_mining.rule_miner.data_to_sequence.dats import Dats

class TestDats(unittest.TestCase):

    def test_mine_rules(self):
        generator = DatasetGenerator(nr_of_categorical_features=2,
                                     nr_of_numerical_features=3,
                                     nr_of_categories=5,
                                     nr_of_instances=120,
                                     random=Random(42),
                                     max_rule_depth=2,
                                     nr_of_sequence_symbols=7)
        dataset, _ = generator.generate()

        miner = Dats()
        mined_rules = miner.mine_rules(dataset)
        self.assertIsNotNone(mined_rules)

        target_sequence = mined_rules.execute(next(iter(dataset)))
        self.assertIsNotNone(target_sequence)

    def test_mine_rules_on_simple_dataset(self):
        dataset = TargetSequenceDataset(['color'],['size'])
        for i in range(40):
            dataset.add_instance(
                TargetSequenceInstance(i, {'color': 'red', 'size': 100}, ['A', 'B']))
        for j in range(40, 60):
            dataset.add_instance(
                TargetSequenceInstance(j, {'color': 'blue', 'size': 100}, ['A', 'C']))

        miner = Dats()
        mined_rules = miner.mine_rules(dataset)

        for instance in dataset:
            self.assertEqual(
                instance.get_target_sequence(),
                tuple(mined_rules.execute(instance)))

    def test_mine_rules_on_simple_dataset_with_filter(self):
        dataset = TargetSequenceDataset(['color'],['size'])
        for i in range(40):
            dataset.add_instance(
                TargetSequenceInstance(i, {'color': 'red', 'size': 100}, ['A', 'B']))
        for j in range(40, 60):
            dataset.add_instance(
                TargetSequenceInstance(j, {'color': 'blue', 'size': 100}, ['A', 'C']))

        miner = Dats(min_relative_sequence_frequency=0.6)
        mined_rules = miner.mine_rules(dataset)

        for instance in dataset:
            self.assertEqual(('A', 'B'), tuple(mined_rules.execute(instance)))


if __name__ == '__main__':
    unittest.main()