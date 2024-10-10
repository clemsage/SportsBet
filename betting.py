import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import predictions


class BettingStrategy(object):
    def __init__(
            self,
            results_predictor: 'predictions.ResultsPredictor',
            evaluate_heuristics: bool = True,
            **kwargs):
        """
        :param results_predictor: Model predicting the match results
        :param evaluate_heuristics: Whether to compare the betting performance when using the provided predictor with
        other baseline predictors
        :param kwargs: Parsed main file arguments
        """
        self.initial_bankroll = kwargs['initial_bankroll']
        print('\nInitial bankroll: %f' % self.initial_bankroll)

        self.betting_platform = kwargs['betting_platform']
        print('Bet platform: %s' % self.betting_platform)

        self.stake_per_bet = kwargs['stake_per_bet']
        print('Stake per bet: %f' % self.stake_per_bet)

        self.do_value_betting = kwargs['do_value_betting']
        print('Bet only on EV+ results: %s' % ('True' if self.do_value_betting else 'False'))

        self.value_betting_on_all_results = kwargs['value_betting_on_all_results']

        self.results_predictor = results_predictor
        self.all_predictors = {self.results_predictor.model_name: self.results_predictor}
        if evaluate_heuristics:
            for baseline in self.results_predictor.baselines:
                self.all_predictors[baseline.__class__.__name__] = baseline

        self.analyze_betting_platforms_margins = kwargs['analyze_betting_platforms_margins']
        # TODO: Evaluate the impact of the betting platforms margins on the gambler's bankroll

        self.total_bet_amount = {predictor: 0 for predictor in self.all_predictors}
        self.bankroll = {predictor: self.initial_bankroll for predictor in self.all_predictors}
        self.bankroll_over_time = {}

    def apply(self, dataset: pd.DataFrame, matches: pd.DataFrame, verbose=False):
        """
        :param dataset: Features of matches
        :param matches: Raw match data
        :param verbose: Whether to print more details during the function execution, e.g. warnings about matches whose
        outcome was not predicted by the model
        :return:

        Leverage the model predictions to bet on the input matches. Update the bankroll following bet wins and losses.

        If *do_value_betting* argument was passed, the system only bets on match results that are likely to beat the
        betting platform, i.e. the result predicted probability times the bookmaker odds is greater than 1. Otherwise,
        the system places a bet on the most likely result for each match.
        """
        for predictor_name, predictor in self.all_predictors.items():
            with_proba = False
            if predictor_name == 'model':
                _dataset = dataset.dropna()
                if self.do_value_betting:
                    with_proba = True
            else:
                _dataset = dataset

            if not len(_dataset):  # prediction is not possible
                continue

            predictions = predictor.infer(_dataset, with_proba=with_proba)

            for i, match in matches.iterrows():
                if i not in predictions.index:  # TODO: correct indexes if we bet on multiple seasons (not unique !)
                    if verbose:
                        print('For %s, the following match has not been predicted: %s against % s at %s' %
                              (predictor_name, match['HomeTeam'], match['AwayTeam'], match['Date']))
                    continue

                if with_proba and self.do_value_betting:  # Compare predicted probabilities and odds
                    max_value = 0
                    bet_result = None
                    if self.value_betting_on_all_results:
                        game_results = ['H', 'D', 'A']
                    else:
                        game_results = [predictions.loc[i, 'result']]
                    for game_result in game_results:
                        value = predictions.loc[i, game_result] * match[''.join((self.betting_platform, game_result))]
                        if value > 1 and value > max_value:
                            bet_result = game_result
                            max_value = value
                    if bet_result is None:
                        continue
                else:  # Bet on the most probable result
                    bet_result = predictions.loc[i, 'result']

                # Update the bankroll of each predictor following bet wins and losses
                if self.bankroll[predictor_name] < self.stake_per_bet:
                    continue  # ran out of money: the system cannot bet anymore
                self.bankroll[predictor_name] -= self.stake_per_bet
                self.total_bet_amount[predictor_name] += self.stake_per_bet
                if match['FTR'] == bet_result:
                    self.bankroll[predictor_name] += self.stake_per_bet * \
                                                     match[''.join((self.betting_platform, match['FTR']))]

    def record_bankroll(self, date: pd.Timestamp):
        """
        :param date: Date for which to record the predictor bankrolls
        :return:

        Record the predictors bankrolls for downstream performance reporting
        """
        self.bankroll_over_time[date] = {}
        for predictor in self.all_predictors:
            self.bankroll_over_time[date][predictor] = self.bankroll[predictor]

    def display_results(self):
        """
        :return:

        Plot the bankroll over time of all results predictors to assess their financial performance
        """
        sns.set()
        self.bankroll_over_time = pd.DataFrame(self.bankroll_over_time)
        for predictor in self.all_predictors:
            print('Predictor: %s' % predictor)
            print('Total amount bet during the season: %f' % self.total_bet_amount[predictor])
            print('Final bankroll: %f\n' % self.bankroll[predictor])
            self.bankroll_over_time.loc[predictor].plot(label=predictor)
        plt.xlabel('Date')
        plt.ylabel('Bankroll')
        plt.title('Betting performance of ML model and baseline heuristics')
        plt.legend()
        plt.show()
