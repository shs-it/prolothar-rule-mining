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

from prolothar_rule_mining.models.dataset_generator import TargetSequenceDatasetGenerator as DatasetGenerator

from random import Random

class TestDatasetGenerator(unittest.TestCase):

    def test_generate_dataset_without_features(self):
        generator = DatasetGenerator(nr_of_categorical_features=0,
                                     nr_of_numerical_features=0)
        dataset, rule = generator.generate()
        self.assertTrue(dataset is not None)
        self.assertTrue(rule is not None)
        self.assertCountEqual([], dataset.get_attributes())

    def test_generate_dataset_with_features(self):
        generator = DatasetGenerator(nr_of_categorical_features=2,
                                     nr_of_numerical_features=3,
                                     nr_of_categories=5,
                                     nr_of_instances=120,
                                     random=Random(42),
                                     max_rule_depth=2,
                                     nr_of_sequence_symbols=7)
        dataset, rule = generator.generate()
        self.assertTrue(dataset is not None)
        self.assertTrue(rule is not None)
        self.assertEqual(120, len(dataset))
        self.assertEqual(5, dataset.get_nr_of_attributes())
        self.assertFalse(
            dataset.get_set_of_sequence_symbols().difference(set(str(i) for i in range(7)))
        )

    def test_generate_dataset_with_features_graph_model(self):
        generator = DatasetGenerator(nr_of_categorical_features=2,
                                     nr_of_numerical_features=3,
                                     nr_of_categories=5,
                                     nr_of_instances=120,
                                     edge_probability=0.1,
                                     random=Random(42),
                                     model='eventflowgraph',
                                     nr_of_sequence_symbols=7)
        dataset, rule = generator.generate()
        self.assertTrue(dataset is not None)
        self.assertTrue(rule is not None)

        self.assertEqual(120, len(dataset))
        self.assertEqual(5, dataset.get_nr_of_attributes())
        self.assertFalse(
            dataset.get_set_of_sequence_symbols().difference(set(str(i) for i in range(7)))
        )

if __name__ == '__main__':
    unittest.main()