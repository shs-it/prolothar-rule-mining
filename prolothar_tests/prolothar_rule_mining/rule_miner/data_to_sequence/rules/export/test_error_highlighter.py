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

from prolothar_common.models.dataset import TargetSequenceDataset
from prolothar_common.models.dataset.instance import TargetSequenceInstance

from prolothar_rule_mining.rule_miner.data_to_sequence.rules.export.html.error_highlighter import ErrorHighlighter
from prolothar_rule_mining.rule_miner.data_to_sequence.rules.export.html.error_highlighter import RULE_STYLE
from prolothar_rule_mining.rule_miner.data_to_sequence.rules import ListOfRules
from prolothar_rule_mining.rule_miner.data_to_sequence.rules import AppendRule
from prolothar_rule_mining.rule_miner.data_to_sequence.rules import IfThenElseRule
from prolothar_rule_mining.models.conditions import EqualsCondition

class TestErrorHighlighter(unittest.TestCase):

    def test_highlight(self):
        dataset = TargetSequenceDataset(['color'],['size'])
        for i in range(40):
            dataset.add_instance(TargetSequenceInstance(
                i, {'color': 'red', 'size': 100}, ['A', 'B']))
        for j in range(40, 60):
            dataset.add_instance(TargetSequenceInstance(
                j, {'color': 'blue', 'size': 100}, ['A', 'C']))

        error_highlighter = ErrorHighlighter()

        empty_model = ListOfRules()
        actual_html = error_highlighter.highlight(empty_model, dataset)
        expected_html = RULE_STYLE + ('<div class="Rule ListOfRules" id="%d">'
                            '<div class="MissingSymbols" style="background: rgba(255,0,0,1.0)">'
                                '<div class="MissingSymbol">'
                                    '<span class="symbol">A</span>'
                                    '<span class="count">60</span>'
                                '</div>'
                            '</div>'
                            '<div class="MissingSymbols" style="background: rgba(255,0,0,1.0)">'
                                '<div class="MissingSymbol">'
                                    '<span class="symbol">B</span>'
                                    '<span class="count">40</span>'
                                '</div>'
                                '<div class="MissingSymbol">'
                                    '<span class="symbol">C</span>'
                                    '<span class="count">20</span>'
                                '</div>'
                            '</div>'
                         '</div>') % id(empty_model)
        self.assertEqual(expected_html, actual_html)

        only_ab_model = ListOfRules([
            AppendRule('A'),
            AppendRule('B')
        ])
        actual_html = error_highlighter.highlight(only_ab_model, dataset)
        self.assertIn(
            '<div class="Rule AppendRule" id="%d">A</div>' % id(only_ab_model[0]),
            actual_html
        )

        wrong_model_b = ListOfRules([
            AppendRule('A'),
            IfThenElseRule(EqualsCondition(
                dataset.get_attribute_by_name('color'), 'blue'),
                if_branch = ListOfRules([
                    AppendRule('C')
                ]),
                else_branch = ListOfRules([
                ])
            ),
            AppendRule('B')
        ])
        actual_html = error_highlighter.highlight(wrong_model_b, dataset)
        self.assertIn('<span class="count">20</span>', actual_html)

if __name__ == '__main__':
    unittest.main()
