import unittest
import pandas as pd

from main import League, betting_platforms, parse_arguments


class TestMethods(unittest.TestCase):
    def test_game_logic(self):
        """
        :return

         Tests the system's ability to correctly track and update team rankings, points, and goals throughout the
         season (matchday by matchday).

         Verifies that the champion team and its final statistics (e.g., played matches, points, goal difference,
         scored and conceded goals) match the expected values.
         """
        kwargs = parse_arguments(args=['--country', 'France', '--start_season', '18'])
        league = League(
            betting_platforms=betting_platforms,
            **kwargs
        )
        league.seasons[0].run()
        simulated_champion = league.seasons[0].ranking.iloc[0]
        expected_champion = pd.Series({'played_matches': 38, 'points': 91, 'goal_difference': 70, 'scored_goals': 105,
                                       'conceded_goals': 35}, name='Paris SG')
        self.assertEqual(simulated_champion.name, expected_champion.name)
        for idx in expected_champion.index:
            self.assertEqual(simulated_champion[idx], expected_champion[idx], 'Incorrect %s count' % idx)


if __name__ == '__main__':
    unittest.main()
