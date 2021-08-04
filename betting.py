

class BettingStrategy(object):
    def __init__(self, args, results_predictor, evaluate_heuristics=True):
        self.initial_bankroll = args.initial_bankroll
        print('\nInitial bankroll: %f' % self.initial_bankroll)

        self.betting_platform = args.betting_platform
        print('Bet platform: %s' % self.betting_platform)

        self.stake_per_bet = args.stake_per_bet
        print('Stake per bet: %f' % self.stake_per_bet)

        self.do_value_betting = args.do_value_betting
        print('Bet only on EV+ results: %s' % ('True' if self.do_value_betting else 'False'))

        self.value_betting_on_all_results = args.value_betting_on_all_results

        self.results_predictor = results_predictor
        self.all_predictors = {'model': self.results_predictor}
        if evaluate_heuristics:
            for baseline in self.results_predictor.baselines:
                self.all_predictors[baseline.__class__.__name__] = baseline

        self.total_bet_amount = {predictor: 0 for predictor in self.all_predictors}
        self.bankroll = {predictor: self.initial_bankroll for predictor in self.all_predictors}

    def apply(self, dataset, matches, verbose=False):
        for predictor_name, predictor in self.all_predictors.items():
            with_proba = False
            if predictor_name == 'model':
                dataset = dataset.dropna()
                if self.do_value_betting:
                    with_proba = True

            if not len(dataset):  # prediction is not possible
                continue

            predictions = predictor.infer(dataset, with_proba=with_proba)

            for i, match in matches.iterrows():
                if i not in predictions.index:
                    if verbose:
                        print('For %s, the following match has not been predicted: %s against % s at %s' %
                              (predictor_name, match['HomeTeam'], match['AwayTeam'], match['Date']))
                    continue

                if with_proba and self.do_value_betting:  # Compare predicted probabilities and odds
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

                self.bankroll[predictor_name] -= self.stake_per_bet
                self.total_bet_amount[predictor_name] += self.stake_per_bet
                if match['FTR'] == bet_result:
                    self.bankroll[predictor_name] += self.stake_per_bet * \
                                                     match[''.join((self.betting_platform, match['FTR']))]
