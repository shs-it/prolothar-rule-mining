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

from prolothar_common.parallel.multiprocess.multiprocess import MultiprocessComputationEngine

from prolothar_rule_mining.models.dataset_generator import TargetSequenceDatasetGenerator as DatasetGenerator
from prolothar_rule_mining.rule_miner.data_to_sequence.consequence.consequence import ConSequence
from prolothar_rule_mining.rule_miner.data_to_sequence.consequence.event_flow_graph_miner import OracleEventFlowGraphMiner

class TestConSequence(unittest.TestCase):

    def test_mine_rules(self):
        generator = DatasetGenerator(nr_of_categorical_features=2,
                                     nr_of_numerical_features=3,
                                     nr_of_categories=5,
                                     nr_of_instances=120,
                                     random=Random(42),
                                     max_rule_depth=2,
                                     nr_of_sequence_symbols=7)
        dataset, _ = generator.generate()

        miner = ConSequence()
        mined_rules = miner.mine_rules(dataset)
        self.assertIsNotNone(mined_rules)

        for instance in dataset:
            print('=================')
            print(instance.get_target_sequence())
            print(mined_rules.execute(instance))

    def test_mine_rules_parallel(self):
        generator = DatasetGenerator(nr_of_categorical_features=2,
                                     nr_of_numerical_features=3,
                                     nr_of_categories=5,
                                     nr_of_instances=120,
                                     random=Random(42),
                                     max_rule_depth=2,
                                     nr_of_sequence_symbols=7)
        dataset, true_model = generator.generate()

        miner = ConSequence(
            event_flow_graph_miner=OracleEventFlowGraphMiner(true_model.to_eventflow_graph()),
            computation_engine=MultiprocessComputationEngine())
        mined_rules = miner.mine_rules(dataset)
        self.assertIsNotNone(mined_rules)

if __name__ == '__main__':
    unittest.main()