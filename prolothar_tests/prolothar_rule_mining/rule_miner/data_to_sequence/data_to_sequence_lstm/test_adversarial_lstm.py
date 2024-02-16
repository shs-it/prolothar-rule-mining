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
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

from random import Random

from prolothar_rule_mining.rule_miner.data_to_sequence.data_to_sequence_lstm import DataToSequenceAdversarialLstm
from prolothar_rule_mining.models.dataset_generator import TargetSequenceDatasetGenerator as DatasetGenerator

class TestDataToSequenceAdversarialLstm(unittest.TestCase):

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
        data2seq = DataToSequenceAdversarialLstm(
            self.dataset, verbose=True, epochs=50, round_limit=1, patience=10,
            early_stopping=True, optimizer_candidates=['Adam'],
            generator_exclusive_epochs=1)
        mined_rule = data2seq.mine_rules()
        predicted_sequence = mined_rule.execute(next(iter(self.dataset)))
        self.assertIsNotNone(predicted_sequence)
        self.assertIsInstance(predicted_sequence, list)

if __name__ == '__main__':
    unittest.main()