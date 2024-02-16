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

from prolothar_rule_mining.models.dataset_generator import TargetSequenceDatasetGenerator as DatasetGenerator
from prolothar_rule_mining.rule_miner.data_to_sequence.window_classifier import WindowClassifier

class TestWindowClassifier(unittest.TestCase):

    def test_mine_rules(self):
        generator = DatasetGenerator(nr_of_categorical_features=2,
                                     nr_of_numerical_features=3,
                                     nr_of_categories=5,
                                     nr_of_instances=10,
                                     max_rule_length={0: 10, 1: 2},
                                     random=Random(42),
                                     max_rule_depth=1,
                                     nr_of_sequence_symbols=20)
        dataset, _ = generator.generate()

        miner = WindowClassifier(nr_of_past_events_as_features=5)
        mined_rules = miner.mine_rules(dataset)
        self.assertIsNotNone(mined_rules)
        instance = next(iter(dataset))
        self.assertIsNotNone(mined_rules.execute(instance))

if __name__ == '__main__':
    unittest.main()