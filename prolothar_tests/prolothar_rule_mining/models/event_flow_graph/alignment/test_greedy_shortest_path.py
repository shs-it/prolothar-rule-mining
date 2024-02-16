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

from prolothar_rule_mining.models.event_flow_graph import EventFlowGraph
from prolothar_rule_mining.models.event_flow_graph.alignment.greedy_shortest_path import GreedyShortestPath
from prolothar_rule_mining.models.event_flow_graph.alignment.alignment import Alignment
from prolothar_rule_mining.models.event_flow_graph.alignment.heuristics import ReachabilityHeuristic
from prolothar_rule_mining.models.event_flow_graph.alignment.heuristics import ProductNetHeuristic

class TestAStar(unittest.TestCase):

    def setUp(self):
        self.graph = EventFlowGraph()

        self.node_a_1 = self.graph.add_node('A')
        self.node_a_2 = self.graph.add_node('A')
        self.node_a_3 = self.graph.add_node('A')
        self.node_b = self.graph.add_node('B')
        self.node_b_2 = self.graph.add_node('B')
        self.node_c = self.graph.add_node('C')
        self.node_c_2 = self.graph.add_node('C')
        self.node_d = self.graph.add_node('D')

        self.graph.add_edge(self.graph.source, self.node_b)
        self.graph.add_edge(self.graph.source, self.node_c)
        self.graph.add_edge(self.graph.source, self.node_b_2)
        self.graph.add_edge(self.node_b, self.node_a_1)
        self.graph.add_edge(self.node_b_2, self.node_a_3)
        self.graph.add_edge(self.node_c, self.node_d)
        self.graph.add_edge(self.node_d, self.node_a_2)
        self.graph.add_edge(self.node_a_1, self.graph.sink)
        self.graph.add_edge(self.node_a_2, self.graph.sink)
        self.graph.add_edge(self.node_a_3, self.node_c_2)
        self.graph.add_edge(self.node_c_2, self.graph.sink)

    def test_compute_alignment_bac(self):
        expected_alignment = Alignment()
        expected_alignment.append_sync_move(self.node_b_2, 0)
        expected_alignment.append_sync_move(self.node_a_3, 1)
        expected_alignment.append_sync_move(self.node_c_2, 2)
        expected_alignment.append_sync_move(self.graph.sink, 3)

        alignment_finder = GreedyShortestPath(self.graph, ReachabilityHeuristic(self.graph))
        actual_alignment = alignment_finder.compute_alignment(['B', 'A', 'C'])
        self.assertEqual(expected_alignment, actual_alignment)

    def test_compute_alignment_empty_sequence(self):
        alignment_finder = GreedyShortestPath(self.graph, ReachabilityHeuristic(self.graph))
        actual_alignment = alignment_finder.compute_alignment([])
        expected_alignment = Alignment()
        expected_alignment.append_model_move(self.node_b)
        expected_alignment.append_model_move(self.node_a_1)
        expected_alignment.append_sync_move(self.graph.sink, 0)
        self.assertEqual(expected_alignment, actual_alignment)

    def test_compute_alignment_bca(self):
        #the reachability heuristic steers us into the wrong direction
        alignment_finder = GreedyShortestPath(self.graph, ReachabilityHeuristic(self.graph))
        actual_alignment = alignment_finder.compute_alignment(['B', 'C', 'A'])
        expected_alignment = Alignment()
        expected_alignment.append_sync_move(self.node_b_2, 0)
        expected_alignment.append_model_move(self.node_a_3)
        expected_alignment.append_sync_move(self.node_c_2, 1)
        expected_alignment.append_log_move(2)
        expected_alignment.append_sync_move(self.graph.sink, 3)
        self.assertEqual(expected_alignment, actual_alignment)

        alignment_finder = GreedyShortestPath(self.graph, ProductNetHeuristic(self.graph))
        actual_alignment = alignment_finder.compute_alignment(['B', 'C', 'A'])
        expected_alignment = Alignment()
        expected_alignment.append_sync_move(self.node_b, 0)
        expected_alignment.append_log_move(1)
        expected_alignment.append_sync_move(self.node_a_1, 2)
        expected_alignment.append_sync_move(self.graph.sink, 3)
        self.assertEqual(expected_alignment, actual_alignment)

    def test_with_connection_from_source_to_sink(self):
        graph = EventFlowGraph()

        node_a = self.graph.add_node('A')

        graph.add_edge(graph.source, node_a)
        graph.add_edge(graph.source, graph.sink)
        graph.add_edge(node_a, graph.sink)

        alignment_finder = GreedyShortestPath(graph, ReachabilityHeuristic(graph))
        actual_alignment = alignment_finder.compute_alignment(['A'])
        expected_alignment = Alignment()
        expected_alignment.append_sync_move(node_a, 0)
        expected_alignment.append_sync_move(graph.sink, 1)
        self.assertEqual(expected_alignment, actual_alignment)

if __name__ == '__main__':
    unittest.main()