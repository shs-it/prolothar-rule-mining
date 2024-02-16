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
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

from prolothar_common.models.eventlog import EventLog, Trace, Event

from prolothar_rule_mining.rule_miner.next_event_prediction.deep_generator import DeepGenerator
from prolothar_rule_mining.rule_miner.next_event_prediction.eventlog_iterator import eventlog_iterator

class TestDeepGenerator(unittest.TestCase):

    def test_on_simple_dataset(self):
        dataset = EventLog()
        id_generator = iter(range(100000))
        for _ in range(800):
            dataset.add_trace(Trace(next(id_generator), [
                Event('A', {'color': 'blue', 'size': random.randint(0, 20)}),
                Event('B')])
            )
            dataset.add_trace(Trace(next(id_generator), [
                Event('A', {'color': 'red', 'size': random.randint(30, 50)}),
                Event('C')])
            )

        predictor = DeepGenerator(
            ['color'], ['size'], nr_of_epochs=20, layer_size=4,
            trace_features_embedding_size=2, event_features_embedding_size=2,
            activity_embedding_size=2, patience=50)
        predictor.train(dataset)

        for prefix, next_event in eventlog_iterator(dataset):
            self.assertIsInstance(predictor.predict(prefix), str)

    def test_on_simple_dataset_with_multiple_categorical_trace_attributes(self):
        dataset = EventLog()
        id_generator = iter(range(100000))
        for _ in range(800):
            dataset.add_trace(Trace(next(id_generator), [
                Event('A', {'shape': random.choice(['circle', 'rect'])}),
                Event('B')],
                {'color': 'blue', 'size': random.choice(['small', 'large'])}
            ))
            dataset.add_trace(Trace(next(id_generator), [
                Event('A'),
                Event('C')],
                {'color': 'red', 'size': random.choice(['small', 'large'])}
            ))

        predictor = DeepGenerator(
            ['color', 'shape', 'size'], [], nr_of_epochs=20, layer_size=4,
            trace_features_embedding_size=2, event_features_embedding_size=2,
            activity_embedding_size=2, patience=50)
        predictor.train(dataset)

        for prefix, next_event in eventlog_iterator(dataset):
            self.assertIsInstance(predictor.predict(prefix), str)

if __name__ == '__main__':
    unittest.main()