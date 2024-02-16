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
"""
This package contains trivial baseline models. Any decent method should be
much better than these baselines
"""

from prolothar_rule_mining.rule_miner.next_event_prediction.trivial.majority_event_predictor import MajorityEventPredictor
from prolothar_rule_mining.rule_miner.next_event_prediction.trivial.prefix_aware_majority_predictor import PrefixAwareMajorityPredictor