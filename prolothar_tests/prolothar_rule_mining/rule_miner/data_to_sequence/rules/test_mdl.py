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

from prolothar_common.models.dataset import TargetSequenceDataset as Dataset
from prolothar_common.models.dataset.instance import TargetSequenceInstance as Instance

from prolothar_rule_mining.rule_miner.data_to_sequence.rules import ListOfRules
from prolothar_rule_mining.rule_miner.data_to_sequence.rules import AppendRule
from prolothar_rule_mining.rule_miner.data_to_sequence.rules import AppendSubsequenceRule
from prolothar_rule_mining.rule_miner.data_to_sequence.rules import IfThenElseRule
from prolothar_rule_mining.models.conditions import EqualsCondition
from prolothar_rule_mining.models.conditions import LessThanCondition
from prolothar_rule_mining.models.conditions import LessOrEqualCondition
from prolothar_rule_mining.models.dataset_generator import TargetSequenceDatasetGenerator as DatasetGenerator
from prolothar_rule_mining.rule_miner.data_to_sequence.rules.mdl import compute_mdl
from prolothar_rule_mining.rule_miner.data_to_sequence.rules.mdl import compute_mdl_of_instance
from prolothar_rule_mining.rule_miner.data_to_sequence.rules.mdl import EditDistanceCache

class TestMdl(unittest.TestCase):

    def test_on_simple_dataset(self):
        dataset = Dataset(['color'],['size'])
        dataset.add_instance(
                    Instance(1, {'color': 'red', 'size': 100}, ['A', 'B']))
        dataset.add_instance(
                    Instance(2, {'color': 'blue', 'size': 100}, ['A', 'C']))

        rule = ListOfRules()
        rule.append_rule(AppendRule('A'))
        bc_rule = IfThenElseRule(EqualsCondition(
                dataset.get_attribute_by_name('color'), 'red'))
        bc_rule.get_if_branch().append_rule(AppendRule('B'))
        bc_rule.get_else_branch().append_rule(AppendRule('C'))
        rule.append_rule(bc_rule)

        mdl = compute_mdl(dataset, rule)
        self.assertGreater(mdl, 0.0)

    def test_expected_relative_ordering_on_simple_dataset(self):
        dataset = Dataset(['color'],['size'])
        for i in range(40):
            dataset.add_instance(
                Instance(i, {'color': 'red', 'size': 100}, ['A', 'B']))
        for j in range(40, 60):
            dataset.add_instance(
                Instance(j, {'color': 'blue', 'size': 100}, ['A', 'C']))

        print('empty_model')
        empty_model = ListOfRules()
        mdl_empty_model = compute_mdl(dataset, empty_model, verbose=True)

        print('only_ab_model')
        only_ab_model = ListOfRules([
            AppendRule('A'),
            AppendRule('B')
        ])
        mdl_only_ab_model = compute_mdl(dataset, only_ab_model, verbose=True)

        print('only_a_model')
        only_a_model = ListOfRules([
            AppendRule('A')
        ])
        mdl_only_a_model = compute_mdl(dataset, only_a_model, verbose=True)

        print('partial_model')
        partial_model = ListOfRules([
            AppendRule('A'),
            IfThenElseRule(EqualsCondition(
                dataset.get_attribute_by_name('color'), 'red'),
                if_branch = ListOfRules([
                    AppendRule('B')
                ]),
            )
        ])
        mdl_partial_model= compute_mdl(dataset, partial_model, verbose=True)

        print('complete_model')
        complete_model = ListOfRules([
            AppendRule('A'),
            IfThenElseRule(EqualsCondition(
                dataset.get_attribute_by_name('color'), 'red'),
                if_branch = ListOfRules([
                    AppendRule('B')
                ]),
                else_branch = ListOfRules([
                    AppendRule('C')
                ])
            )
        ])
        mdl_complete_model = compute_mdl(dataset, complete_model, verbose=True)

        wrong_model_b = ListOfRules([
            AppendRule('A'),
            IfThenElseRule(EqualsCondition(
                dataset.get_attribute_by_name('color'), 'blue'),
                if_branch = ListOfRules([
                    AppendRule('C')
                ]),
                else_branch = ListOfRules([
                ])
            ),
            AppendRule('B')
        ])
        mdl_wrong_model_b = compute_mdl(dataset, wrong_model_b, verbose=False)

        wrong_model_c = ListOfRules([
            AppendRule('A'),
            IfThenElseRule(EqualsCondition(
                dataset.get_attribute_by_name('color'), 'red'),
                if_branch = ListOfRules([
                    AppendRule('B')
                ]),
                else_branch = ListOfRules([
                ])
            ),
            AppendRule('C')
        ])
        mdl_wrong_model_c = compute_mdl(dataset, wrong_model_c, verbose=False)

        self.assertLess(mdl_only_a_model, mdl_empty_model)
        self.assertLess(mdl_complete_model, mdl_empty_model)
        self.assertLess(mdl_complete_model, mdl_only_ab_model)
        self.assertLess(mdl_complete_model, mdl_partial_model)
        self.assertLess(mdl_partial_model, mdl_only_ab_model)
        self.assertLess(mdl_complete_model, mdl_wrong_model_b)
        self.assertLess(mdl_complete_model, mdl_wrong_model_c)

    def test_singletons_should_improve_model(self):
        generator = DatasetGenerator(nr_of_categorical_features=2,
                                nr_of_numerical_features=3,
                                nr_of_categories=5,
                                nr_of_instances=120,
                                random=Random(42),
                                max_rule_depth=2,
                                nr_of_sequence_symbols=20,
                                max_rule_length={
                                    0: 20,
                                    1: 3,
                                    2: 1
                                })
        dataset, rule = generator.generate()

        mdl_true_model = compute_mdl(dataset, rule)
        mdl_empty_model = compute_mdl(dataset, ListOfRules())

        model = ListOfRules([AppendRule('13')])
        mdl_singleton_model = compute_mdl(dataset, model)

        model2 = ListOfRules([AppendSubsequenceRule(['13'])])
        mdl_singleton_model2 = compute_mdl(dataset, model2)

        self.assertLess(mdl_singleton_model, mdl_empty_model)
        self.assertLess(mdl_singleton_model2, mdl_empty_model)
        self.assertLess(mdl_true_model, mdl_singleton_model)

    def test_if_else_should_improve_model(self):
        generator = DatasetGenerator(nr_of_categorical_features=2,
                                nr_of_numerical_features=3,
                                nr_of_categories=5,
                                nr_of_instances=120,
                                random=Random(42),
                                max_rule_depth=2,
                                nr_of_sequence_symbols=20,
                                max_rule_length={
                                    0: 20,
                                    1: 3,
                                    2: 1
                                })
        dataset, rule = generator.generate()

        model = ListOfRules([
            AppendRule('13'),
            AppendRule('15'),
            AppendRule('10'),
            AppendRule('2'),
            AppendRule('6'),
            AppendRule('15'),
            AppendRule('9'),
            AppendRule('9'),
            AppendRule('17'),
            AppendRule('8'),
            AppendRule('1'),
            AppendRule('15'),
            AppendRule('1'),
            AppendRule('2'),
            AppendRule('4'),
            AppendRule('18'),
            AppendRule('16'),
            AppendRule('19'),
            AppendRule('6'),
            AppendRule('8')
        ])
        mdl_model = compute_mdl(dataset, model)

        improved_model = ListOfRules([
            AppendRule('13'),
            AppendRule('15'),
            AppendRule('10'),
            AppendRule('2'),
            AppendRule('6'),
            AppendRule('15'),
            AppendRule('9'),
            AppendRule('9'),
            AppendRule('17'),
            AppendRule('8'),
            AppendRule('1'),
            AppendRule('15'),
            AppendRule('1'),
            AppendRule('2'),
            IfThenElseRule(
                LessThanCondition(
                    dataset.get_attribute_by_name('num_feature_1'),
                    0.6649294342127715
                ),
                AppendRule('4')
            ),
            AppendRule('18'),
            AppendRule('16'),
            AppendRule('19'),
            AppendRule('6'),
            AppendRule('8')
        ])
        mdl_improved_model = compute_mdl(dataset, improved_model)
        mdl_optimal_model = compute_mdl(dataset, rule)

        self.assertLess(mdl_improved_model, mdl_model)
        self.assertLess(mdl_optimal_model, mdl_improved_model)


    def test_on_simple_dataset_with_edit_distance_cache(self):
        dataset = Dataset(['color'],['size'])
        dataset.add_instance(
                    Instance(1, {'color': 'red', 'size': 100}, ['A', 'B']))
        dataset.add_instance(
                    Instance(2, {'color': 'blue', 'size': 100}, ['A', 'C']))

        rule = ListOfRules()
        rule.append_rule(AppendRule('A'))
        bc_rule = IfThenElseRule(EqualsCondition(
                dataset.get_attribute_by_name('color'), 'red'))
        bc_rule.get_if_branch().append_rule(AppendRule('B'))
        bc_rule.get_else_branch().append_rule(AppendRule('C'))
        rule.append_rule(bc_rule)

        mdl_without_cache = compute_mdl(dataset, rule)
        edit_distance_cache = EditDistanceCache(2)
        mdl_with_cache_first_run = compute_mdl(
            dataset, rule, edit_distance_cache=edit_distance_cache)
        mdl_with_cache_second_run = compute_mdl(
            dataset, rule, edit_distance_cache=edit_distance_cache)
        self.assertEqual(mdl_without_cache, mdl_with_cache_first_run)
        self.assertEqual(mdl_without_cache, mdl_with_cache_second_run)

    def test_merge_of_two_following_ifs_should_improve_score(self):
        dataset = Dataset(['color'],['size'])
        for i in range(40):
            dataset.add_instance(
                Instance(i, {'color': 'blue', 'size': 100}, ['A', 'B', 'C', 'D']))
        for i in range(40, 60):
            dataset.add_instance(
                Instance(i, {'color': 'red', 'size': 120}, ['A', 'B']))
        for i in range(60, 70):
            dataset.add_instance(
                Instance(i, {'color': 'blue', 'size': 180}, ['B', 'E']))

        model_1 = ListOfRules([
            AppendRule('A'),
            AppendRule('B'),
            IfThenElseRule(
                LessOrEqualCondition(dataset.get_attribute_by_name('size'), 100),
                if_branch = ListOfRules([
                    AppendRule('C')
                ])),
            IfThenElseRule(
                EqualsCondition(dataset.get_attribute_by_name('color'), 'blue'),
                if_branch = ListOfRules([
                    AppendRule('D')
                ])),
        ])
        mdl_1 = compute_mdl(dataset, model_1)

        model_2 = ListOfRules([
            AppendRule('A'),
            AppendRule('B'),
            IfThenElseRule(
                LessOrEqualCondition(dataset.get_attribute_by_name('size'), 100),
                if_branch = ListOfRules([
                    AppendRule('C'),
                    AppendRule('D')
                ]))
        ])
        mdl_2 = compute_mdl(dataset, model_2)

        self.assertLess(mdl_2, mdl_1)

if __name__ == '__main__':
    unittest.main()