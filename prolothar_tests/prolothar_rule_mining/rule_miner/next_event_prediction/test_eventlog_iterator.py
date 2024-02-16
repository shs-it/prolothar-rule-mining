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

from prolothar_rule_mining.rule_miner.next_event_prediction.eventlog_iterator import eventlog_iterator


class TestEventLogIterator(unittest.TestCase):

    def test_iterate(self):
        dataset = EventLog()
        dataset.traces = [
            Trace(0, [Event('A'), Event('B')]),
            Trace(1, [Event('A'), Event('C')]),
        ]

        expected_iterated_instances = [
            (Trace('0-0', [Event('ε')]), 'A'),
            (Trace('0-1', [Event('ε'), Event('A')]), 'B'),
            (Trace('0-2', [Event('ε'), Event('A'), Event('B')]), 'ω'),
            (Trace('1-0', [Event('ε')]), 'A'),
            (Trace('1-1', [Event('ε'), Event('A')]), 'C'),
            (Trace('1-2', [Event('ε'), Event('A'), Event('C')]), 'ω'),
        ]

        actual_iterated_instances = list(eventlog_iterator(dataset))

        self.assertEqual(expected_iterated_instances, actual_iterated_instances)

if __name__ == '__main__':
    unittest.main()
