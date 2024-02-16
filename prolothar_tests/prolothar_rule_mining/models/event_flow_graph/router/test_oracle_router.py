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

from prolothar_common.models.dataset.instance import TargetSequenceInstance

from prolothar_rule_mining.models.event_flow_graph import EventFlowGraph
from prolothar_rule_mining.models.event_flow_graph.router.oracle_router import LocalOracleRouter
from prolothar_rule_mining.models.event_flow_graph.router.oracle_router import GlobalOracleRouter

class TestOracleRouter(unittest.TestCase):

    def test_oracle_router(self):
        graph = EventFlowGraph()
        self.assertEqual(0, graph.get_nr_of_nodes())

        node_a_1 = graph.add_node('A')
        node_a_2 = graph.add_node('A')
        node_b = graph.add_node('B')
        node_c = graph.add_node('C')
        node_d = graph.add_node('D')

        graph.add_edge(graph.source, node_b)
        graph.add_edge(graph.source, node_c)
        graph.add_edge(node_b, node_a_1)
        graph.add_edge(node_c, node_d)
        graph.add_edge(node_d, node_a_2)
        graph.add_edge(node_a_1, graph.sink)
        graph.add_edge(node_a_2, graph.sink)

        graph.plot(show=False, filepath='temp')

        global_router = GlobalOracleRouter(graph)
        local_router = LocalOracleRouter(global_router, graph.source)

        self.assertEqual(node_b, local_router(TargetSequenceInstance(0, {}, ['B', 'A'])))

if __name__ == '__main__':
    unittest.main()