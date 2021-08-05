"""
Main file for the Sports Bet project
"""
import argparse
import pandas as pd
import numpy as np

from game import League
from predictions import ResultsPredictor
from betting import BettingStrategy


pd.options.display.width = 0
np.seterr(divide='ignore', invalid='ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Game
    parser.add_argument(
        '--country',
        required=True,
        type=str,
        help="Name of the league's country, among {Italy, France, Spain, England, Germany}."
    )

    parser.add_argument(
        '--division',
        type=str,
        default="1",
        help="Division level, e.g. 1 for Premier League in England."
    )

    parser.add_argument(
        '--start_season',
        required=True,
        type=int,
        help="Two digit year indicating the first season analyzed, e.g. 04 for the 2004/2005 season."
    )

    parser.add_argument(
        '--end_season',
        required=False,
        type=int,
        help="Two digit year indicating the last season analyzed, e.g. 05 for the 2004/2005 season. If not"
             "specified, we consider the season indicated by the start_season argument."
    )

    # Betting
    parser.add_argument(
        '--betting_platform',
        required=True,
        type=str,
        default="B365",
        help="Ticker of the betting platform, among {B365, BW, IW, PS, WH, VC}. "
             "Some platform may not be available for the chosen league."
    )

    parser.add_argument(
        '--initial_bankroll',
        type=float,
        default=100,
        help="Initial amount allowed for betting."
    )

    parser.add_argument(
        '--stake_per_bet',
        type=float,
        default=1.0,
        help="Stake for each bet."
    )

    parser.add_argument(
        '--do_value_betting',
        action="store_true",
        default=False,
        help="If true, bet only if the expected value of a result is positive, "
             "i.e. if its predicted probability is above the inverse of its odd from the betting "
             "platform. If false, we always bet on the most likely predicted result, whatever its odd."
    )

    parser.add_argument(
        '--value_betting_on_all_results',
        action="store_true",
        default=False,
        help="If true, perform value betting on the three results, i.e. victory of the home, away team "
             "and draw. If false, perform value betting only on the most likely predicted result."
    )

    # Features
    parser.add_argument(
        '--match_history_length',
        default=None,
        type=int,
        help="Number of previous matches for each facing team included in the model features."
             "If the value k is provided, then we consider the last k home and the last k away matches for each team."
             "If not specified, we do not consider the game history."
    )

    parser.add_argument(
        '--add_match_scores',
        action="store_true",
        default=False,
        help="Besides the results, add the match scores to the features."
    )

    parser.add_argument(
        '--number_previous_direct_confrontations',
        default=3,
        type=int,
        help="Use the last k direct confrontations between the two teams as features."
    )

    parser.add_argument(
        '--match_results_encoding',
        default='points',
        type=str,
        help="In the feature vectors, encode the result of a match either with a categorical value among 'Win', "
             "'Draw' and 'Loose' or with a numerical value, i.e. the number of won points which is respectively 3, 1"
             " and 0. The value of this argument is chosen among {'categorical', 'points'}"
    )

    # Model
    parser.add_argument(
        '--model_name',
        default='LogisticRegression',
        type=str,
        help='Chosen predictive model following the scikit-learn nomenclature.'
             'Supported values are {LogisticRegression, MLPClassifier, DecisionTreeClassifier, RandomForestClassifier}'
    )

    parser.add_argument(
        '--config_name',
        default=None,
        type=str,
        help="Model configuration name or path. By default, search for the file $model_name.json in the models folder"
    )

    args = parser.parse_args()

    # Set the game
    league = League(args)
    league.run()

    results_predictor = ResultsPredictor(league, args)
    results_predictor.train()
    results_predictor.eval()

    betting_strategy = BettingStrategy(args, results_predictor)
    league.seasons[-1].run(betting_strategy)
    betting_strategy.display_results()
