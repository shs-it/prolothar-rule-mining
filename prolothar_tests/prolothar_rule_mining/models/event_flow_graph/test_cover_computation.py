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

from prolothar_rule_mining.models.event_flow_graph import EventFlowGraph
from prolothar_rule_mining.models.event_flow_graph.cover import compute_cover
from prolothar_rule_mining.models.dataset_generator import TargetSequenceDatasetGenerator as DatasetGenerator

class TestCoverComputation(unittest.TestCase):

    def test_compute_cover_on_empty_model(self):
        generator = DatasetGenerator(nr_of_categorical_features=2,
                                     nr_of_numerical_features=0,
                                     nr_of_categories=5,
                                     nr_of_instances=120,
                                     random=Random(42),
                                     max_rule_depth=2,
                                     nr_of_sequence_symbols=100)
        dataset, _ = generator.generate()

        event_flow_graph = EventFlowGraph()
        event_flow_graph.add_edge(event_flow_graph.source, event_flow_graph.sink)

        cover = compute_cover(dataset, event_flow_graph)
        self.assertIsNotNone(cover)

        model_counts = cover.count_model_codes()
        self.assertEqual(0, model_counts.redundant_events)
        self.assertEqual(120, model_counts.matched_events)
        self.assertGreater(model_counts.missed_events, 0)

if __name__ == '__main__':
    unittest.main()