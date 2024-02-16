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

from prolothar_rule_mining.rule_miner.data_to_sequence.event_classifier import EventClassifier
from prolothar_rule_mining.rule_miner.classification.rce import ReliableRuleMiner
from prolothar_rule_mining.models.dataset_generator import TargetSequenceDatasetGenerator as DatasetGenerator

class TestEventClassifier(unittest.TestCase):

    def setUp(self):
        generator = DatasetGenerator(nr_of_categorical_features=2,
                                     nr_of_numerical_features=3,
                                     nr_of_categories=5,
                                     nr_of_instances=120,
                                     random=Random(42),
                                     max_rule_depth=2,
                                     nr_of_sequence_symbols=7)
        self.dataset, _ = generator.generate()

    def test_mine_rules(self):
        mined_rule = EventClassifier(ReliableRuleMiner()).mine_rules(self.dataset)
        predicted_sequence = mined_rule.execute(next(iter(self.dataset)))
        self.assertIsNotNone(predicted_sequence)
        self.assertIsInstance(predicted_sequence, tuple)

if __name__ == '__main__':
    unittest.main()