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

from prolothar_common.models.dataset.instance import Instance
from prolothar_common.models.dataset.attributes import CategoricalAttribute
from prolothar_common.models.dataset.attributes import NumericalAttribute
from prolothar_rule_mining.models.conditions import EqualsCondition
from prolothar_rule_mining.models.conditions import GreaterThanCondition
from prolothar_rule_mining.models.conditions import LessOrEqualCondition
from prolothar_rule_mining.models.conditions import LessThanCondition
from prolothar_rule_mining.models.conditions import NotCondition
from prolothar_rule_mining.models.conditions import AndCondition
from prolothar_rule_mining.models.conditions import OrCondition
from prolothar_rule_mining.models.conditions import InCondition

class TestConditions(unittest.TestCase):

    def setUp(self):
        self.color_attribute = CategoricalAttribute('Color', {'red', 'green', 'blue'})
        self.size_attribute = NumericalAttribute('Size', list(range(120, 210)))
        self.weight_attribute = NumericalAttribute('Weight', list(range(0, 150)))

    def test_equals(self):
        condition = EqualsCondition(self.color_attribute, 'red')
        self.assertTrue(condition.check_instance(Instance(0, {'Color': 'red'})))
        self.assertFalse(condition.check_instance(Instance(0, {'Color': 'blue'})))

    def test_in(self):
        condition = InCondition(self.color_attribute, {'red', 'green'})
        self.assertTrue(condition.check_instance(Instance(0, {'Color': 'red'})))
        self.assertFalse(condition.check_instance(Instance(0, {'Color': 'blue'})))
        self.assertTrue(condition.check_instance(Instance(0, {'Color': 'green'})))

    def test_greater_than(self):
        condition = GreaterThanCondition(self.size_attribute, 150)
        self.assertTrue(condition.check_instance(Instance(0, {'Size': 180})))
        self.assertTrue(condition.check_instance(Instance(0, {'Size': 151})))
        self.assertFalse(condition.check_instance(Instance(0, {'Size': 150})))
        self.assertFalse(condition.check_instance(Instance(0, {'Size': 149})))
        self.assertFalse(condition.check_instance(Instance(0, {'Size': 140})))

    def test_less_than(self):
        condition = LessThanCondition(self.size_attribute, 150)
        self.assertFalse(condition.check_instance(Instance(0, {'Size': 180})))
        self.assertFalse(condition.check_instance(Instance(0, {'Size': 151})))
        self.assertFalse(condition.check_instance(Instance(0, {'Size': 150})))
        self.assertTrue(condition.check_instance(Instance(0, {'Size': 140})))
        self.assertTrue(condition.check_instance(Instance(0, {'Size': 149})))

    def test_not(self):
        condition = NotCondition(
                GreaterThanCondition(self.size_attribute, 150))
        self.assertFalse(condition.check_instance(Instance(0, {'Size': 180})))
        self.assertFalse(condition.check_instance(Instance(0, {'Size': 151})))
        self.assertTrue(condition.check_instance(Instance(0, {'Size': 150})))
        self.assertTrue(condition.check_instance(Instance(0, {'Size': 140})))
        self.assertTrue(condition.check_instance(Instance(0, {'Size': 149})))

    def test_and(self):
        condition = AndCondition([
            GreaterThanCondition(self.size_attribute, 150),
            EqualsCondition(self.color_attribute, 'red')
        ])
        self.assertFalse(condition.check_instance(Instance(0, {'Size': 180, 'Color': 'blue'})))
        self.assertTrue(condition.check_instance(Instance(0, {'Size': 151, 'Color': 'red'})))
        self.assertFalse(condition.check_instance(Instance(0, {'Size': 150, 'Color': 'blue'})))
        self.assertFalse(condition.check_instance(Instance(0, {'Size': 140, 'Color': 'red'})))
        self.assertFalse(condition.check_instance(Instance(0, {'Size': 149, 'Color': 'blue'})))


    def test_or(self):
        condition = OrCondition([
            GreaterThanCondition(self.size_attribute, 150),
            EqualsCondition(self.color_attribute, 'red')
        ])
        self.assertTrue(condition.check_instance(Instance(0, {'Size': 180, 'Color': 'blue'})))
        self.assertTrue(condition.check_instance(Instance(0, {'Size': 151, 'Color': 'red'})))
        self.assertFalse(condition.check_instance(Instance(0, {'Size': 150, 'Color': 'blue'})))
        self.assertTrue(condition.check_instance(Instance(0, {'Size': 140, 'Color': 'red'})))
        self.assertFalse(condition.check_instance(Instance(0, {'Size': 149, 'Color': 'blue'})))

    def test_compress_and_condition(self):
        condition = AndCondition([
            AndCondition([
                EqualsCondition(self.color_attribute, 'blue'),
                GreaterThanCondition(self.size_attribute, 10),
            ]),
            GreaterThanCondition(self.size_attribute, 20)
        ])
        expected_compressed_condition = AndCondition([
            EqualsCondition(self.color_attribute, 'blue'),
            GreaterThanCondition(self.size_attribute, 20)
        ])
        actual_compressed_condition = condition.compress()
        self.assertEqual(expected_compressed_condition, actual_compressed_condition)

    def test_compress_or_condition(self):
        condition = OrCondition([
            OrCondition([
                EqualsCondition(self.color_attribute, 'blue'),
                GreaterThanCondition(self.size_attribute, 10),
            ]),
            GreaterThanCondition(self.size_attribute, 20)
        ])
        expected_compressed_condition = OrCondition([
            EqualsCondition(self.color_attribute, 'blue'),
            GreaterThanCondition(self.size_attribute, 10)
        ])
        actual_compressed_condition = condition.compress()
        self.assertEqual(expected_compressed_condition, actual_compressed_condition)

    def test_compress_and_or_condition(self):
        condition = AndCondition([
            OrCondition([
                EqualsCondition(self.color_attribute, 'blue'),
                GreaterThanCondition(self.size_attribute, 10),
            ]),
            GreaterThanCondition(self.size_attribute, 10)
        ])
        expected_compressed_condition = GreaterThanCondition(self.size_attribute, 10)
        actual_compressed_condition = condition.compress()
        self.assertEqual(expected_compressed_condition, actual_compressed_condition)

    def test_compress_or_and_condition(self):
        condition = OrCondition([
            AndCondition([
                EqualsCondition(self.color_attribute, 'blue'),
                GreaterThanCondition(self.size_attribute, 10),
            ]),
            GreaterThanCondition(self.size_attribute, 10)
        ])
        expected_compressed_condition = GreaterThanCondition(self.size_attribute, 10)
        actual_compressed_condition = condition.compress()
        self.assertEqual(expected_compressed_condition, actual_compressed_condition)


    def test_compress_nested_or_condition(self):
        condition = OrCondition([
            OrCondition([
                EqualsCondition(self.color_attribute, 'blue'),
                GreaterThanCondition(self.size_attribute, 10),
            ]),
            EqualsCondition(self.color_attribute, 'red')
        ])
        expected_compressed_condition = OrCondition([
            EqualsCondition(self.color_attribute, 'blue'),
            GreaterThanCondition(self.size_attribute, 10),
            EqualsCondition(self.color_attribute, 'red'),
        ])
        actual_compressed_condition = condition.compress()
        self.assertCountEqual(
            expected_compressed_condition.get_conditions(),
            actual_compressed_condition.get_conditions())

if __name__ == '__main__':
    unittest.main()