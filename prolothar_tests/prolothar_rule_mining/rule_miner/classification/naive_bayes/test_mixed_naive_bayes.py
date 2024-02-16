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
from prolothar_rule_mining.rule_miner.classification.naive_bayes import MixedNaiveBayesClassifier

class TestMixedNaiveBayesClassifier(unittest.TestCase):

    def test_mine_rules_differentiate_three_simple_classes(self):
        dataset = ClassificationDataset(['category'], ['size'])
        for i in range(25):
            dataset.add_instance(ClassificationInstance(
                'A%d' % i, {'category': 'A', 'size': 5}, 'A'))
        for i in range(50):
            dataset.add_instance(ClassificationInstance(
                'B%d' % i, {'category': 'B', 'size': 17}, 'B'))
        for i in range(21):
            dataset.add_instance(ClassificationInstance(
                'Csmall%d' % i, {'category': 'C', 'size': 7}, 'CS'))
        for i in range(20):
            dataset.add_instance(ClassificationInstance(
                'Clarge%d' % i, {'category': 'C', 'size': 21}, 'CL'))

        miner = MixedNaiveBayesClassifier()
        rule = miner.mine_rules(dataset)
        self.assertTrue(rule is not None)

        for instance in dataset:
            predicted_class = rule.predict(instance)
            self.assertEqual(instance.get_class(), predicted_class)
            predicted_class = max(rule.predict_proba(instance).items(), key=lambda x: x[1])[0]
            self.assertEqual(instance.get_class(), predicted_class)

if __name__ == '__main__':
    unittest.main()