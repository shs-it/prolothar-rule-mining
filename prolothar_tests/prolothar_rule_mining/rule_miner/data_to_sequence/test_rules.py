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
from prolothar_rule_mining.rule_miner.data_to_sequence.rules import AppendRule
from prolothar_rule_mining.rule_miner.data_to_sequence.rules import IfThenElseRule
from prolothar_rule_mining.rule_miner.data_to_sequence.rules import ListOfRules

from prolothar_rule_mining.models.event_flow_graph import EventFlowGraph

class TestRules(unittest.TestCase):

    def test_execute(self):
        color_attribute = CategoricalAttribute('Color', {'red', 'green', 'blue'})
        size_attribute = NumericalAttribute('Size', list(range(120, 210)))
        rule = IfThenElseRule(EqualsCondition(color_attribute, 'red'))
        rule.get_if_branch().append_rule(AppendRule('A'))
        else_rule = IfThenElseRule(GreaterThanCondition(size_attribute, 150))
        else_rule.get_if_branch().append_rule(AppendRule('B'))
        rule.get_else_branch().append_rule(else_rule)

        self.assertListEqual(['A'], rule.execute(Instance(0, {'Color': 'red'})))
        self.assertListEqual(['B'], rule.execute(Instance(0, {'Color': 'blue', 'Size': 160})))
        self.assertListEqual([], rule.execute(Instance(0, {'Color': 'blue', 'Size': 140})))

    def test_compute_mdl(self):
        color_attribute = CategoricalAttribute('Color', {'red', 'green', 'blue'})
        size_attribute = NumericalAttribute('Size', list(range(120, 210)))
        rule = IfThenElseRule(EqualsCondition(color_attribute, 'red'))
        rule.get_if_branch().append_rule(AppendRule('A'))
        else_rule = IfThenElseRule(GreaterThanCondition(size_attribute, 150))
        else_rule.get_if_branch().append_rule(AppendRule('B'))
        rule.get_else_branch().append_rule(else_rule)

        self.assertTrue(rule.compute_mdl({'A', 'B'}, 2) is not None)
        self.assertGreater(rule.compute_mdl({'A', 'B'}, 2), 0)

    def test_count_nr_of_terms(self):
        color_attribute = CategoricalAttribute('Color', {'red', 'green', 'blue'})
        size_attribute = NumericalAttribute('Size', list(range(120, 210)))
        rule = IfThenElseRule(EqualsCondition(color_attribute, 'red'))
        rule.get_if_branch().append_rule(AppendRule('A'))
        else_rule = IfThenElseRule(GreaterThanCondition(size_attribute, 150))
        else_rule.get_if_branch().append_rule(AppendRule('B'))
        rule.get_else_branch().append_rule(else_rule)
        self.assertEqual(2, rule.count_nr_of_terms())

    def test_to_html(self):
        color_attribute = CategoricalAttribute('Color', {'red', 'green', 'blue'})
        size_attribute = NumericalAttribute('Size', list(range(120, 210)))
        rule = IfThenElseRule(EqualsCondition(color_attribute, 'red'))
        append_a = AppendRule('A')
        rule.get_if_branch().append_rule(append_a)
        else_rule = IfThenElseRule(GreaterThanCondition(size_attribute, 150))
        append_b = AppendRule('B')
        else_rule.get_if_branch().append_rule(append_b)
        rule.get_else_branch().append_rule(else_rule)
        actual_html = rule.to_html()
        self.assertIsNotNone(actual_html)
        expected_html = (
            '<div class="Rule IfThenElseRule">'
                '<div class="If">If</div>'
                '<div class="Condition AttributeCondition">'
                    '<span class="Attribute">Color</span>'
                    '<span class="Operator">=</span>'
                    '<span class="Value">"red"</span>'
                '</div>'
                '<div class="ifbranch">'
                    '<div id="%d" class="Rule ListOfRules">'
                        '<div id="%d" class="Rule AppendRule">A</div>'
                    '</div>'
                '</div>'
                '<div class="Else">Else</div>'
                '<div class="elsebranch">'
                    '<div id="%d" class="Rule ListOfRules">'
                        '<div class="Rule IfThenElseRule">'
                            '<div class="If">If</div>'
                            '<div class="Condition AttributeCondition">'
                                '<span class="Attribute">Size</span>'
                                '<span class="Operator">&gt;</span>'
                                '<span class="Value">150</span>'
                            '</div>'
                            '<div class="ifbranch">'
                                '<div id="%d" class="Rule ListOfRules">'
                                    '<div id="%d" class="Rule AppendRule">B</div>'
                                '</div>'
                            '</div>'
                            '<div class="elsebranch">'
                                '<div id="%d" class="Rule ListOfRules">'
                                '</div>'
                            '</div>'
                        '</div>'
                    '</div>'
                '</div>'
            '</div>'
        ) % (
            id(rule.get_if_branch()), id(append_a), id(rule.get_else_branch()),
            id(else_rule.get_if_branch()), id(append_b),
            id(else_rule.get_else_branch())
        )
        self.assertEqual(expected_html, actual_html)

    def test_to_graphviz(self):
        color_attribute = CategoricalAttribute('Color', {'red', 'green', 'blue'})
        size_attribute = NumericalAttribute('Size', list(range(120, 210)))
        rule = IfThenElseRule(EqualsCondition(color_attribute, 'red'))
        append_a = AppendRule('A')
        rule.get_if_branch().append_rule(append_a)
        else_rule = IfThenElseRule(GreaterThanCondition(size_attribute, 150))
        append_b = AppendRule('B')
        else_rule.get_if_branch().append_rule(append_b)
        rule.get_else_branch().append_rule(else_rule)
        rule = ListOfRules([AppendRule('START'), rule, AppendRule('END')])

        digraph = rule.to_graphviz()
        self.assertIsNotNone(digraph)

    def test_to_eventflow_graph(self):
        color_attribute = CategoricalAttribute('Color', {'red', 'green', 'blue'})
        size_attribute = NumericalAttribute('Size', list(range(120, 210)))
        rule = IfThenElseRule(EqualsCondition(color_attribute, 'red'))
        append_a = AppendRule('A')
        rule.get_if_branch().append_rule(append_a)
        else_rule = IfThenElseRule(GreaterThanCondition(size_attribute, 150))
        append_b = AppendRule('B')
        else_rule.get_if_branch().append_rule(append_b)
        rule.get_else_branch().append_rule(else_rule)
        rule = ListOfRules([AppendRule('START'), rule, AppendRule('END')])

        expected_graph = EventFlowGraph()
        node_start = expected_graph.add_node('START')
        node_end = expected_graph.add_node('END')
        node_a = expected_graph.add_node('A')
        node_b = expected_graph.add_node('B')
        expected_graph.add_edge(expected_graph.source, node_start)
        expected_graph.add_edge(node_start, node_end)
        expected_graph.add_edge(node_start, node_a)
        expected_graph.add_edge(node_start, node_b)
        expected_graph.add_edge(node_a, node_end)
        expected_graph.add_edge(node_b, node_end)
        expected_graph.add_edge(node_end, expected_graph.sink)

        actual_graph = rule.to_eventflow_graph()

        self.assertEqual(actual_graph.get_nr_of_nodes(), expected_graph.get_nr_of_nodes())
        self.assertEqual(actual_graph.get_nr_of_edges(), expected_graph.get_nr_of_edges())

        for actual_node, expected_node in zip(
                sorted(actual_graph.nodes(), key=lambda node: node.event),
                sorted(expected_graph.nodes(), key=lambda node: node.event)):
            self.assertEqual(actual_node.event, expected_node.event)
            actual_next_events = [node.event for node in actual_node.children]
            expected_next_events = [node.event for node in expected_node.children]
            self.assertCountEqual(actual_next_events, expected_next_events)

if __name__ == '__main__':
    unittest.main()