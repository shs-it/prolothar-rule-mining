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

class TestEventFlowGraph(unittest.TestCase):

    def test_create_graph(self):
        graph = EventFlowGraph()
        self.assertEqual(0, graph.get_nr_of_nodes())

        node_a_1 = graph.add_node('A')
        self.assertIsNotNone(node_a_1)
        self.assertEqual(1, graph.get_nr_of_nodes())

        node_a_2 = graph.add_node('A')
        self.assertIsNotNone(node_a_2)
        self.assertEqual(2, graph.get_nr_of_nodes())

        graph.add_edge(graph.source, node_a_1)
        graph.add_edge(graph.source, node_a_2)
        graph.add_edge(node_a_1, graph.sink)
        graph.add_edge(node_a_2, graph.sink)
        self.assertEqual(4, graph.get_nr_of_edges())

    def test_prevent_connections_to_source_and_from_sink(self):
        graph = EventFlowGraph()
        self.assertEqual(0, graph.get_nr_of_nodes())

        node_a_1 = graph.add_node('A')

        with self.assertRaises(ValueError):
            graph.add_edge(node_a_1, graph.source)

        with self.assertRaises(ValueError):
            graph.add_edge(graph.sink, node_a_1)

    def test_find_shortest_path_to_event(self):
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

        path = graph.find_shortest_path_to_event(graph.source, 'A')
        self.assertIsNotNone(path)
        self.assertListEqual([graph.source, node_b, node_a_1], path)

    def test_find_shortest_paths_to_event(self):
        graph = EventFlowGraph()
        self.assertEqual(0, graph.get_nr_of_nodes())

        node_a_1 = graph.add_node('A')
        node_a_2 = graph.add_node('A')
        node_b_1 = graph.add_node('B')
        node_b_2 = graph.add_node('B')

        graph.add_edge(graph.source, node_a_1)
        graph.add_edge(graph.source, node_a_2)
        graph.add_edge(node_a_1, node_b_1)
        graph.add_edge(node_a_2, node_b_2)
        graph.add_edge(node_b_1, graph.sink)
        graph.add_edge(node_b_2, graph.sink)

        expected_paths = [
            [graph.source, node_a_1],
            [graph.source, node_a_2],
        ]
        actual_paths = graph.find_shortest_paths_to_event(graph.source, 'A')
        self.assertCountEqual(expected_paths, actual_paths)

        expected_paths = [
            [graph.source, node_a_1, node_b_1],
            [graph.source, node_a_2, node_b_2],
        ]
        actual_paths = graph.find_shortest_paths_to_event(graph.source, 'B')
        self.assertCountEqual(expected_paths, actual_paths)

    def test_find_chains(self):
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

        expected_chains = [
            [node_b, node_a_1],
            [node_c, node_d, node_a_2]
        ]

        actual_chains = list(graph.find_chains())

        self.assertCountEqual(expected_chains, actual_chains)

    def test_find_chains_2(self):
        graph = EventFlowGraph()

        node_a = graph.add_node('A')
        node_b = graph.add_node('B')
        node_c = graph.add_node('C')

        graph.add_edge(graph.source, node_a)
        graph.add_edge(node_a, node_b)
        graph.add_edge(node_b, node_c)
        graph.add_edge(node_c, graph.sink)
        graph.add_edge(node_a, node_c)

        expected_chains = [
            [graph.source, node_a],
            [node_c, graph.sink]
        ]

        actual_chains = list(graph.find_chains())
        self.assertCountEqual(expected_chains, actual_chains)

    def test_contains_cycle(self):
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

        self.assertFalse(graph.contains_cycle())

        graph.add_edge(node_a_2, node_c)
        self.assertTrue(graph.contains_cycle())

    def test_to_json(self):
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

        json = graph.to_json()
        self.assertEqual(graph, EventFlowGraph.from_json(json))

    def test_remove_illegal_source_nodes(self):
        graph = EventFlowGraph()
        node_a_1 = graph.add_node('A')
        node_a_2 = graph.add_node('A')
        node_b = graph.add_node('B')
        node_c = graph.add_node('C')
        node_d = graph.add_node('D')

        graph.add_edge(graph.source, node_b)
        graph.add_edge(node_b, node_a_1)
        graph.add_edge(node_c, node_d)
        graph.add_edge(node_d, node_a_2)
        graph.add_edge(node_a_1, graph.sink)
        graph.add_edge(node_a_2, graph.sink)

        expected_graph = graph.copy()
        expected_graph.remove_node(expected_graph.get_node_by_id(node_c.node_id))
        expected_graph.remove_node(expected_graph.get_node_by_id(node_a_2.node_id))
        expected_graph.remove_node(expected_graph.get_node_by_id(node_d.node_id))

        graph.remove_illegal_source_nodes()

        self.assertEqual(expected_graph, graph)

if __name__ == '__main__':
    unittest.main()