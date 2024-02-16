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

from prolothar_common.models.dataset.attributes import CategoricalAttribute
from prolothar_common.models.dataset.attributes import NumericalAttribute

from prolothar_rule_mining.models.conditions import EqualsCondition
from prolothar_rule_mining.models.conditions import LessOrEqualCondition

from prolothar_rule_mining.rule_miner.data_to_sequence.rules.event_flow_graph_rule import EventFlowGraph
from prolothar_rule_mining.rule_miner.data_to_sequence.rules.event_flow_graph_rule import EventFlowGraphRule
from prolothar_rule_mining.rule_miner.data_to_sequence.rules.event_flow_graph_rule import RuledRouter
from prolothar_rule_mining.rule_miner.classification.rules import FirstFiringRuleModel
from prolothar_rule_mining.rule_miner.classification.rules import ListOfRules
from prolothar_rule_mining.rule_miner.classification.rules import IfThenElseRule
from prolothar_rule_mining.rule_miner.classification.rules import ReturnClassRule

class TestEventFlowGraphRule(unittest.TestCase):

    def test_to_odm_xml(self):
        graph = EventFlowGraph()
        node_a_1 = graph.add_node('A')
        node_a_2 = graph.add_node('A')
        node_b = graph.add_node('B')
        node_c = graph.add_node('C')
        node_d = graph.add_node('D')
        node_e = graph.add_node('E')

        graph.add_edge(graph.source, node_b)
        graph.add_edge(graph.source, node_c)
        graph.add_edge(graph.source, node_e)
        graph.add_edge(node_b, node_a_1)
        graph.add_edge(node_c, node_d)
        graph.add_edge(node_d, node_a_2)
        graph.add_edge(node_a_1, graph.sink)
        graph.add_edge(node_a_2, graph.sink)
        graph.add_edge(node_e, graph.sink)

        color_attribute = CategoricalAttribute('Color', {'red', 'green', 'blue'})
        size_attribute = NumericalAttribute('Size', list(range(120, 210)))

        node_router_table = {
            graph.source: RuledRouter(
                {
                    node_b.node_id: node_b,
                    node_c.node_id: node_c,
                    node_e.node_id: node_e
                },
                FirstFiringRuleModel(ListOfRules([
                    IfThenElseRule(
                        EqualsCondition(color_attribute, 'red'),
                        if_branch=ListOfRules([ReturnClassRule(str(node_b.node_id))])
                    ),
                    IfThenElseRule(
                        LessOrEqualCondition(size_attribute, 150),
                        if_branch=ListOfRules([ReturnClassRule(str(node_c.node_id))])
                    ),
                    ReturnClassRule(str(node_e.node_id))
                ]))
            )
        }

        model = EventFlowGraphRule(graph, node_router_table)

        with open('prolothar_tests/resources/expected_odm_xml.xml') as f:
            expected_xml = f.read()

        operator_formatting_dict = {
            '=': 'is equal to',
            '<=': 'is at most'
        }

        self.assertEqual(
            expected_xml.replace(' ', ''),
            model.to_odm_xml(
                attribute_formatter=lambda a: f'the {a.lower()} of input',
                operator_formatter=operator_formatting_dict.get,
                categorical_value_formatter=lambda s: '"' + s + '"'
            ).replace(' ', '')
        )

if __name__ == '__main__':
    unittest.main()