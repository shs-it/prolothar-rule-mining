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

from prolothar_common.models.eventlog import EventLog, Trace, Event
from prolothar_rule_mining.models.event_flow_graph import EventFlowGraph

from prolothar_rule_mining.rule_miner.next_event_prediction.trivial import PrefixAwareMajorityPredictor
from prolothar_rule_mining.rule_miner.next_event_prediction.eventlog_iterator import eventlog_iterator

class TestPrefixAwareMajorityPredictor(unittest.TestCase):

    def test_on_simple_dataset(self):
        dataset = EventLog()
        dataset.traces = [
            Trace(0, [Event('A', {'color': 'blue'}), Event('B')]),
            Trace(1, [Event('A', {'color': 'blue'}), Event('B')]),
            Trace(2, [Event('A', {'color': 'blue'}), Event('B')]),
            Trace(3, [Event('A', {'color': 'blue'}), Event('B')]),
            Trace(4, [Event('A', {'color': 'blue'}), Event('B')]),
            Trace(5, [Event('A', {'color': 'blue'}), Event('B')]),
            Trace(6, [Event('A', {'color': 'blue'}), Event('B')]),
            Trace(7, [Event('A', {'color': 'blue'}), Event('B')]),
            Trace(8, [Event('A', {'color': 'red'}), Event('C')]),
            Trace(9, [Event('A', {'color': 'red'}), Event('C')]),
        ]

        predictor = PrefixAwareMajorityPredictor()
        predictor.train(dataset)

        sink_event = EventFlowGraph().sink.event
        for prefix, next_event in eventlog_iterator(dataset):
            if len(prefix) == 1:
                self.assertEqual(predictor.predict(prefix), 'A')
            elif len(prefix) == 2:
                self.assertEqual(predictor.predict(prefix), 'B')
            elif len(prefix) == 3:
                self.assertEqual(predictor.predict(prefix), sink_event)

if __name__ == '__main__':
    unittest.main()
