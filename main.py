"""
Main file for the Sports Bet project
"""
import pandas as pd
import numpy as np
import datetime
from copy import deepcopy
from sklearn import preprocessing, compose
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import urllib


pd.options.display.width = 0
np.seterr(divide='ignore', invalid='ignore')

#################
# DO NOT CHANGE #
base_path_data = 'https://www.football-data.co.uk/mmz4281'
#################

country = 'France'
division = '1'
seasons = '0519'  # list of 4 digit strings, e.g. 1819 for the 2018/2019 season
# or just a 4 digit string, e.g. 1019 for considering the season 10/11 until the season 18/19 included
bet_platform = 'B365'  # among B365, BW, IW, PS, WH, VC. Some may not be available for the chosen league
initial_bankroll = 100  # in €
only_EV_plus_bets = True

###########################
# Features for prediction #
use_last_k_matches = {'Home': 3, 'Away': 3}  # None to disable
use_last_k_matches_scores = False  # If False, use only game results
###########################

if isinstance(seasons, str):
    start_season = int(seasons[:2])
    end_season = int(seasons[2:])
    seasons = []
    for year in range(start_season, end_season):
        seasons.append('%s%s' % (str(year % 100).zfill(2), str((year+1) % 100).zfill(2)))


class League(object):
    def __init__(self, country, division, seasons):
        self.country = country
        print('Country: %s' % country)
        self.division = division
        print('Division: %s' % division)
        id_country = country[0].upper()
        if country.lower() == 'spain':
            id_country = 'SP'
        elif country.lower() == 'germany':
            id_country = 'D'
        self.name = id_country + division
        self.seasons = [Season(self.name, season) for season in seasons]
        print('Seasons: %s' % seasons)
        self.datasets = {}

    def run(self):
        for season in self.seasons:
            season.run()
            self.datasets[season.name] = season.dataset


class Season(object):
    def __init__(self, league_name, name):
        self.league_name = league_name
        self.name = name
        data_url = '/'.join((base_path_data, name, league_name + '.csv'))
        try:  # TODO: Retrieve the csv if locally available
            self.matches = pd.read_csv(data_url, sep=',', encoding='mbcs')
        except urllib.error.HTTPError:
            print('The following data URL seems incorrect: %s' % data_url)
        self.matches = self.matches.dropna(how='all')

        # sort matches by chronological order
        def normalize_year(year):  # fix for 2017/2018 French 1st league having DD/MM/YY format instead of DD/MM/YYYY
            if len(year) == 2:
                current_year = int(str(datetime.datetime.now().year)[-2:])
                if int(year) <= current_year:
                    year = '20' + year
                else:
                    year = '19' + year
            return year
        self.matches['day'] = self.matches['Date'].apply(lambda x: x.split('/')[0])
        self.matches['month'] = self.matches['Date'].apply(lambda x: x.split('/')[1])
        self.matches['year'] = self.matches['Date'].apply(lambda x: normalize_year(x.split('/')[2]))
        self.matches['Date'] = self.matches.apply(lambda df: '/'.join((df['day'], df['month'], df['year'])), axis=1)
        self.matches['Date'] = pd.to_datetime(self.matches['Date'], format='%d/%m/%Y')
        self.matches.sort_values(by=['Date'], inplace=True)

        self._matches = None
        self.teams = None
        self.ranking = None
        self.dataset = None
        self.reset_statistics()

    def reset_statistics(self):
        team_names = self.matches['HomeTeam'].unique()
        self.teams = {team_name: Team(team_name) for team_name in team_names}
        self.ranking = self.get_ranking()
        self.dataset = []

    def update_statistics(self, played_matches):
        for stat in ['FTR', 'FTHG', 'FTAG']:
            assert stat in played_matches, '%s statistics must be available' % stat

        for _, match in played_matches.iterrows():
            assert match['FTR'] in ['H', 'D', 'A'], '%s unknown match result' % match['FTR']
            for home_or_away in ['Home', 'Away']:
                self.teams[match['%sTeam' % home_or_away]].update(match, home_or_away)
        self.ranking = self.get_ranking()

    def get_ranking(self):
        ranking_props = ['name', 'played_matches', 'points', 'goal_difference', 'scored_goals', 'conceded_goals']
        ranking = pd.DataFrame([{key: value for key, value in vars(team).items() if key in ranking_props}
                                for team in self.teams.values()])
        ranking.set_index('name', inplace=True)
        ranking.sort_values(['points', 'goal_difference'], ascending=False, inplace=True)
        for team in self.teams.values():
            team.ranking = 1 + ranking.index.get_loc(team.name)
        return ranking

    def run(self, betting_strategy=None):
        self.reset_statistics()
        self._matches = deepcopy(self.matches)
        while len(self._matches):
            # Group matches by date
            current_date = self._matches['Date'].iloc[0]
            matches = self._matches.loc[self._matches['Date'] == current_date]
            dataset = []
            for _, match in matches.iterrows():
                dataset.append(self.prepare_example(match))
            dataset = pd.DataFrame(dataset, index=matches.index)
            dataset = dataset.dropna()  # drop the matches with Nan in the features, i.e. usually the first
            # use_last_k_matches game days
            if betting_strategy is not None:
                betting_strategy.apply(dataset, matches)
            self.update_statistics(matches)
            self._matches = self._matches[self._matches['Date'] != current_date]
            self.dataset.append(dataset)
        self.dataset = pd.concat(self.dataset)

    def prepare_example(self, match):
        example = {'result': match['FTR']}  # ground truth

        # Features
        for home_or_away in ['Home', 'Away']:
            team_name = match['%sTeam' % home_or_away]
            team = self.teams[team_name]
            example['%sPlayedMatches'] = team.played_matches
            example['%sRanking' % home_or_away] = team.ranking
            example['%sAvgPoints' % home_or_away] = np.divide(team.points, team.played_matches)

            if use_last_k_matches is not None:
                for prev_home_or_away in ['Home', 'Away']:
                    for i in range(1, 1 + use_last_k_matches[prev_home_or_away]):
                        features = ['Res']
                        if use_last_k_matches_scores:
                            features.extend(['FT%sG' % prev_home_or_away[0],
                                             'FT%sG' % ('H' if prev_home_or_away == "Away" else 'A')])

                        for feature in features:
                            key = '%sPrev%s%s%d' % (home_or_away, prev_home_or_away, feature, i)
                            if i <= len(team.last_k_matches[prev_home_or_away]):
                                example[key] = team.last_k_matches[prev_home_or_away][-i][feature]
                            else:
                                example[key] = np.nan
        return example


class Team(object):
    def __init__(self, name):
        self.name = name
        self.played_matches = 0
        self.points = 0
        self.goal_difference = 0
        self.scored_goals = 0
        self.conceded_goals = 0
        self.ranking = None
        self.last_k_matches = {'Home': [], 'Away': []}

    def update(self, match, home_or_away):
        self.played_matches += 1
        if match['FTR'] == home_or_away[0]:
            self.points += 3
            match['Res'] = 'W'  # win
        elif match['FTR'] == 'D':
            self.points += 1
            match['Res'] = 'D'
        else:
            match['Res'] = 'L'  # loose
        self.scored_goals += match['FT%sG' % home_or_away[0]]
        self.conceded_goals += match['FT%sG' % ('A' if home_or_away == 'Home' else 'H')]
        self.goal_difference = self.scored_goals - self.conceded_goals
        if use_last_k_matches is not None:
            self.last_k_matches[home_or_away].append(match)
            self.last_k_matches[home_or_away] = self.last_k_matches[home_or_away][-use_last_k_matches[home_or_away]:]


def dataset_preprocessing(dataset, label_encoder=None, feature_preprocessor=None):
    Y = dataset['result']
    X = dataset.drop('result', axis='columns')

    # Transform categorical labels (H, D, A) into numerical values
    if label_encoder is None:
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(Y)
    Y = label_encoder.transform(Y)

    if feature_preprocessor is None:
        # print(X.info())
        feature_preprocessor = compose.ColumnTransformer(transformers=[
            ('cat', preprocessing.OneHotEncoder(), compose.make_column_selector(dtype_include="object")),
            ('num', preprocessing.StandardScaler(), compose.make_column_selector(dtype_exclude="object"))
        ])
        feature_preprocessor.fit(X)
    X = feature_preprocessor.transform(X)

    return X, Y, label_encoder, feature_preprocessor


class ResultsPredictor(object):
    def __init__(self, league):
        # TODO: Model selection and hyperparameter tuning
        self.model = LogisticRegression(C=1e5)
        self.model = MLPClassifier(hidden_layer_sizes=32, alpha=1, verbose=False, max_iter=10000, random_state=44)
        # self.model = DecisionTreeClassifier()
        # self.model = RandomForestClassifier()
        print('Model: %s' % self.model.__class__.__name__)
        self.league = league
        self.training_dataset, self.test_dataset = self.split_train_test_sets()

    def split_train_test_sets(self):
        """
        Train on the N-1 first seasons and evaluate on the last season
        """
        training_dataset = []
        for i in range(len(self.league.seasons) - 1):
            training_dataset.append(self.league.datasets[self.league.seasons[i].name])
        training_dataset = pd.concat(training_dataset)
        test_dataset = self.league.datasets[self.league.seasons[-1].name]
        return training_dataset, test_dataset

    def train(self):
        X, Y, self.label_encoder, self.feature_preprocessor = dataset_preprocessing(self.training_dataset)
        print('Feature set size: %d' % X.shape[1])
        print('\nTraining of the model on %d samples...' % len(self.training_dataset))
        self.model.fit(X, Y)
        Y_pred = self.model.predict(X)
        print('Training accuracy of the model: %.3f' % accuracy_score(Y, Y_pred))

    def eval(self):
        print('Evaluation of the model on %d samples...' % len(self.test_dataset))
        X, Y, _, _ = dataset_preprocessing(self.test_dataset, self.label_encoder, self.feature_preprocessor)
        Y_pred = self.model.predict(X)
        print('Test accuracy of the model: %.3f' % accuracy_score(Y, Y_pred))

        # Comparison to two baseline models
        # 1) The home team always win
        Y_pred = self.label_encoder.transform(['H'] * len(Y_pred))
        print('Test accuracy of the heuristic "The home team always wins": %.3f' % accuracy_score(Y, Y_pred))

        # 2) The best ranked team always wins
        Y_pred = pd.Series(['A'] * len(Y_pred), index=self.test_dataset.index)
        home_team_has_best_ranking = self.test_dataset['HomeRanking'] < self.test_dataset['AwayRanking']
        Y_pred[home_team_has_best_ranking] = 'H'
        Y_pred = self.label_encoder.transform(Y_pred)
        print('Test accuracy of the heuristic "The best ranked team always wins": %.3f' % accuracy_score(Y, Y_pred))

    def infer(self, dataset, with_proba=False):
        X, _, _, _ = dataset_preprocessing(dataset, self.label_encoder, self.feature_preprocessor)
        if with_proba:
            Y_pred = self.model.predict_proba(X)
            Y_pred = pd.DataFrame(Y_pred, columns=self.label_encoder.classes_, index=dataset.index)
            Y_pred['result'] = Y_pred.idxmax(axis=1)
        else:
            Y_pred = self.model.predict(X)
            Y_pred = self.label_encoder.inverse_transform(Y_pred)
            Y_pred = pd.DataFrame(Y_pred, columns=['result'], index=dataset.index)
        return Y_pred


class BettingStrategy(object):
    def __init__(self, initial_bankroll, results_predictor, bet_platform, bet_per_match=1):
        self.initial_bankroll = initial_bankroll
        print('\nInitial bankroll: %f' % initial_bankroll)
        self.bankroll = initial_bankroll
        self.results_predictor = results_predictor
        self.bet_platform = bet_platform
        print('Bet platform: %s' % self.bet_platform)
        self.bet_per_match = bet_per_match
        self.total_bet_amount = 0

    def apply(self, dataset, matches):
        if len(dataset):  # if prediction is possible
            predictions = self.results_predictor.infer(dataset, with_proba=True if only_EV_plus_bets else False)
            for i, match in matches.iterrows():
                if i not in predictions.index:
                    print('The following match has not been predicted: %s against % s at %s' %
                          (match['HomeTeam'], match['AwayTeam'], match['Date']))
                    continue

                if only_EV_plus_bets:
                    if predictions.loc[i, predictions.loc[i, 'result']] < \
                            (1 / match[''.join((bet_platform, predictions.loc[i, 'result']))]):
                        continue

                self.bankroll -= self.bet_per_match
                self.total_bet_amount += self.bet_per_match
                if match['FTR'] == predictions.loc[i, 'result']:
                    self.bankroll += self.bet_per_match * match[''.join((bet_platform, match['FTR']))]


league = League(country, division, seasons)
league.run()
results_predictor = ResultsPredictor(league)
results_predictor.train()
results_predictor.eval()
betting_strategy = BettingStrategy(initial_bankroll, results_predictor, bet_platform)
league.seasons[-1].run(betting_strategy)
print('Total amount bet during the season: %f' % betting_strategy.total_bet_amount)
print('Final bankroll: %f' % betting_strategy.bankroll)
