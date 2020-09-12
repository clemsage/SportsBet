import unittest
import pandas as pd

from main import League


class TestMethods(unittest.TestCase):
    def test_game_logic(self):
        league = League(country='France', division='1', seasons = ['1819'])
        league.seasons[0].run()
        simulated_champion = league.seasons[0].ranking.iloc[0]
        expected_champion = pd.Series({'played_matches': 38, 'points': 91, 'goal_difference': 70, 'scored_goals': 105,
                                       'conceded_goals': 35}, name='Paris SG')
        self.assertEqual(simulated_champion.name, expected_champion.name)
        for idx in expected_champion.index:
            self.assertEqual(simulated_champion[idx], expected_champion[idx], 'Incorrect %s count' % idx)
