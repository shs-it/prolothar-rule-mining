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

from prolothar_common.models.dataset import TargetSequenceDataset
from prolothar_common.models.dataset.instance import TargetSequenceInstance
from prolothar_rule_mining.models.dataset_generator import TargetSequenceDatasetGenerator as DatasetGenerator
from prolothar_rule_mining.rule_miner.data_to_sequence.consequence.event_flow_graph_miner import SequenceTopDownEventFlowGraphMiner
from prolothar_rule_mining.models.event_flow_graph.alignment.beam_search import BeamSearch
from prolothar_rule_mining.models.event_flow_graph.alignment.heuristics import ReachabilityHeuristic

class TestSequenceTopDownEventFlowGraphMiner(unittest.TestCase):

    def test_mine_event_flow_graph(self):
        generator = DatasetGenerator(nr_of_categorical_features=2,
                                     nr_of_numerical_features=3,
                                     nr_of_categories=5,
                                     nr_of_instances=120,
                                     random=Random(42),
                                     max_rule_depth=2,
                                     nr_of_sequence_symbols=7)
        dataset, rule = generator.generate()

        event_flow_graph = SequenceTopDownEventFlowGraphMiner(
            patience=10, exact_mdl=True,
            alignment_finder_factory_score=lambda g:
            BeamSearch(g, ReachabilityHeuristic(g), 5)
        ).mine_event_flow_graph(dataset)
        self.assertIsNotNone(event_flow_graph)
        self.assertGreater(len(event_flow_graph.source.children), 0)
        self.assertGreater(len(event_flow_graph.sink.parents), 0)
        self.assertIsNotNone(event_flow_graph.find_shortest_path(event_flow_graph.source, event_flow_graph.sink))
        self.assertFalse(event_flow_graph.contains_cycle())

    def test_mine_event_flow_graph_with_estimated_mdl(self):
        generator = DatasetGenerator(nr_of_categorical_features=2,
                                     nr_of_numerical_features=3,
                                     nr_of_categories=5,
                                     nr_of_instances=120,
                                     random=Random(42),
                                     max_rule_depth=2,
                                     nr_of_sequence_symbols=7)
        dataset, rule = generator.generate()

        event_flow_graph = SequenceTopDownEventFlowGraphMiner(
            patience=10, exact_mdl=False,
            alignment_finder_factory_score=lambda g:
            BeamSearch(g, ReachabilityHeuristic(g), 5)
        ).mine_event_flow_graph(dataset)
        self.assertIsNotNone(event_flow_graph)
        self.assertGreater(len(event_flow_graph.source.children), 0)
        self.assertGreater(len(event_flow_graph.sink.parents), 0)
        self.assertIsNotNone(event_flow_graph.find_shortest_path(event_flow_graph.source, event_flow_graph.sink))
        self.assertFalse(event_flow_graph.contains_cycle())

    def test_reconstruct_one_chain(self):
        dataset = TargetSequenceDataset([], [])
        for i in range(20):
            dataset.add_instance(TargetSequenceInstance(i, {}, ['A', 'B', 'C']))

        event_flow_graph = SequenceTopDownEventFlowGraphMiner().mine_event_flow_graph(dataset)
        self.assertEqual(3, event_flow_graph.get_nr_of_nodes())
        self.assertEqual(4, event_flow_graph.get_nr_of_edges())

    def test_reconstruct_two_independent_chains(self):
        dataset = TargetSequenceDataset([], [])
        for i in range(20):
            dataset.add_instance(TargetSequenceInstance(i, {}, ['A', 'B', 'C']))
            dataset.add_instance(TargetSequenceInstance(20 + i, {}, ['D', 'E', 'F']))

        event_flow_graph = SequenceTopDownEventFlowGraphMiner().mine_event_flow_graph(dataset)
        self.assertEqual(6, event_flow_graph.get_nr_of_nodes())
        self.assertEqual(8, event_flow_graph.get_nr_of_edges())

    def test_reconstruct_one_chain_with_estimated_mdl(self):
        dataset = TargetSequenceDataset([], [])
        for i in range(20):
            dataset.add_instance(TargetSequenceInstance(i, {}, ['A', 'B', 'C']))

        event_flow_graph = SequenceTopDownEventFlowGraphMiner(exact_mdl=False).mine_event_flow_graph(dataset)
        self.assertEqual(3, event_flow_graph.get_nr_of_nodes())
        self.assertEqual(4, event_flow_graph.get_nr_of_edges())

    def test_reconstruct_two_independent_chains_with_estimated_mdl(self):
        dataset = TargetSequenceDataset([], [])
        for i in range(20):
            dataset.add_instance(TargetSequenceInstance(i, {}, ['A', 'B', 'C']))
            dataset.add_instance(TargetSequenceInstance(20 + i, {}, ['D', 'E', 'F']))

        event_flow_graph = SequenceTopDownEventFlowGraphMiner(exact_mdl=False).mine_event_flow_graph(dataset)

        self.assertEqual(6, event_flow_graph.get_nr_of_nodes())
        self.assertEqual(8, event_flow_graph.get_nr_of_edges())

if __name__ == '__main__':
    unittest.main()