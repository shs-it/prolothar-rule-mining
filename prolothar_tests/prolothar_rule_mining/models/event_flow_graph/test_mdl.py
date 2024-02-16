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
from typing import Tuple

import unittest
from random import Random

from prolothar_common.models.dataset import TargetSequenceDataset
from prolothar_common.models.dataset.instance import TargetSequenceInstance

from prolothar_rule_mining.models.event_flow_graph import EventFlowGraph
from prolothar_rule_mining.models.event_flow_graph.cover import compute_cover
from prolothar_rule_mining.models.event_flow_graph.cover import Cover
from prolothar_rule_mining.models.dataset_generator import TargetSequenceDatasetGenerator as DatasetGenerator

def compute_mdl(graph: EventFlowGraph, dataset: TargetSequenceDataset, verbose=False) -> Tuple[float, Cover]:
    mdl_of_model = graph.compute_mdl(dataset.get_set_of_sequence_symbols())
    cover = compute_cover(dataset, graph)
    mdl_of_data = cover.compute_mdl()
    if verbose:
        print((mdl_of_model, mdl_of_data))
        print(cover.count_model_codes())
    return mdl_of_model + mdl_of_data, cover

class TestMdl(unittest.TestCase):

    def test_compute_mdl(self):
        generator = DatasetGenerator(nr_of_categorical_features=2,
                                     nr_of_numerical_features=0,
                                     nr_of_categories=5,
                                     nr_of_instances=120,
                                     random=Random(42),
                                     max_rule_depth=2,
                                     nr_of_sequence_symbols=100)
        dataset, _ = generator.generate()

        model = EventFlowGraph()
        model.add_edge(model.source, model.sink)
        mdl_empty_model, cover = compute_mdl(model, dataset, verbose=True)
        model_codes = cover.count_model_codes()
        self.assertGreater(model_codes.missed_events, 0)
        self.assertEqual(model_codes.redundant_events, 0)
        self.assertEqual(model_codes.matched_events, 120)

        node_35 = model.add_node('35')
        model.add_edge(model.source, node_35)
        model.add_edge(node_35, model.sink)
        mdl_node_35, cover = compute_mdl(model, dataset, verbose=True)
        model_codes = cover.count_model_codes()
        self.assertGreater(model_codes.missed_events, 0)
        self.assertEqual(model_codes.redundant_events, 0)
        self.assertGreaterEqual(model_codes.matched_events, 240)
        self.assertLess(mdl_node_35, mdl_empty_model)

        node_40 = model.add_node('40')
        model.add_edge(node_35, node_40)
        model.add_edge(node_40, model.sink)
        mdl_node_40, cover = compute_mdl(model, dataset, verbose=True)
        model_codes = cover.count_model_codes()
        self.assertGreater(model_codes.missed_events, 0)
        self.assertEqual(model_codes.redundant_events, 0)
        self.assertGreater(model_codes.matched_events, 320)
        self.assertLess(mdl_node_40, mdl_node_35)

        node_84 = model.add_node('84')
        model.add_edge(node_40, node_84)
        model.add_edge(node_84, model.sink)
        mdl_node_84, cover = compute_mdl(model, dataset, verbose=True)
        model_codes = cover.count_model_codes()
        self.assertGreater(model_codes.missed_events, 0)
        self.assertGreater(model_codes.matched_events, 420)
        self.assertLess(mdl_node_84, mdl_node_40)

        node_10 = model.add_node('10')
        model.add_edge(node_84, node_10)
        model.add_edge(node_10, model.sink)
        mdl_node_10, _ = compute_mdl(model, dataset, verbose=True)
        self.assertLess(mdl_node_10, mdl_node_84)

    def test_chain_should_be_better_than_empty_model_for_one_sequence(self):
        dataset = TargetSequenceDataset([], [])
        for i in range(20):
            dataset.add_instance(TargetSequenceInstance(i, {}, ['A', 'B', 'C']))

        model = EventFlowGraph()

        edge = model.add_edge(model.source, model.sink)
        empty_model_mdl = compute_mdl(model, dataset)
        model.remove_edge(edge)

        node_a = model.add_node('A')
        node_b = model.add_node('B')
        node_c = model.add_node('C')

        model.add_edge(model.source, node_a)
        model.add_edge(node_a, node_b)
        model.add_edge(node_b, node_c)
        model.add_edge(node_c, model.sink)

        chain_model_mdl = compute_mdl(model, dataset)

        self.assertLess(chain_model_mdl, empty_model_mdl)

if __name__ == '__main__':
    unittest.main()