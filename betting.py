

class BettingStrategy(object):
    def __init__(self, args, results_predictor):
        self.initial_bankroll = args.initial_bankroll
        print('\nInitial bankroll: %f' % self.initial_bankroll)

        self.betting_platform = args.betting_platform
        print('Bet platform: %s' % self.betting_platform)

        self.stake_per_bet = args.stake_per_bet
        print('Stake per bet: %f' % self.stake_per_bet)

        self.do_value_betting = args.do_value_betting
        print('Bet only on EV+ results: %s' % ('True' if self.do_value_betting else 'False'))

        self.value_betting_on_all_results = args.value_betting_on_all_results

        self.total_bet_amount = 0
        self.bankroll = self.initial_bankroll
        self.results_predictor = results_predictor

    def apply(self, dataset, matches, verbose=False):
        if len(dataset):  # if prediction is possible
            predictions = self.results_predictor.infer(dataset, with_proba=True if self.do_value_betting else False)
            for i, match in matches.iterrows():
                if i not in predictions.index:
                    if verbose:
                        print('The following match has not been predicted: %s against % s at %s' %
                              (match['HomeTeam'], match['AwayTeam'], match['Date']))
                    continue

                if self.do_value_betting:  # Compare predicted probabilities and odds of the betting platform
                    max_value = 0
                    bet_result = None
                    game_results = [['H', 'D', 'A'] if self.value_betting_on_all_results else
                                    predictions.loc[i, 'result']]
                    for game_result in game_results:
                        value = predictions.loc[i, game_result] * match[''.join((self.betting_platform, game_result))]
                        if value > 1 and value > max_value:
                            bet_result = game_result
                            max_value = value
                    if bet_result is None:
                        continue
                else:  # Bet on the most probable result
                    bet_result = predictions.loc[i, 'result']

                # TODO: track the evolution of the bankroll over the time
                self.bankroll -= self.stake_per_bet
                self.total_bet_amount += self.stake_per_bet
                if match['FTR'] == bet_result:
                    self.bankroll += self.stake_per_bet * match[''.join((self.betting_platform, match['FTR']))]
