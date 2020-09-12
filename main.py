"""
Main file for the Sports Bet project
"""
import pandas as pd
pd.options.display.width = 0
import numpy as np
import datetime
from copy import deepcopy
from sklearn import preprocessing, compose
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#################
# DO NOT CHANGE #
base_path_data = 'https://www.football-data.co.uk/mmz4281'
#################

country = 'France'
division = '1'
seasons = ['1415', '1516', '1617', '1718', '1819']  # list of 4 digits, e.g. 1819 for the 2018/2019 season

###########################
# Features for prediction #
use_last_k_matches = 5  # None to not use it or integer
###########################


class League(object):
    def __init__(self, country, division, seasons):
        self.country = country
        self.division = division
        self.name = country[0].upper() + division
        self.seasons = [Season(self.name, season) for season in seasons]
        self.datasets = {}

    def run(self):
        for season in self.seasons:
            season.run()
            self.datasets[season.name] = pd.DataFrame(season.dataset)


class Season(object):
    def __init__(self, league_name, name):
        self.league_name = league_name
        self.name = name
        data_url = '/'.join((base_path_data, name, league_name + '.csv'))
        self.matches = pd.read_csv(data_url, sep=',', encoding='mbcs')
        self.matches = self.matches.dropna(how='all')
        self._matches = deepcopy(self.matches)

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

    def run(self):
        while len(self.matches):
            # Group matches by date
            current_date = self.matches['Date'].iloc[0]
            matches = self.matches.loc[self.matches['Date'] == current_date]
            for _, match in matches.iterrows():
                self.dataset.append(self.prepare_example(match))
            self.update_statistics(matches)
            self.matches = self.matches[self.matches['Date'] != current_date]

    def prepare_example(self, match):
        example = {'result': match['FTR']}  # ground truth

        # Features
        for home_or_away in ['Home', 'Away']:
            team_name = match['%sTeam' % home_or_away]
            team = self.teams[team_name]
            example['%sRanking' % home_or_away] = team.ranking
            example['%sAvgPoints' % home_or_away] = np.divide(team.points, team.played_matches)

            if use_last_k_matches is not None:
                for i in range(1, 1 + use_last_k_matches):
                    if i <= len(team.last_k_matches):
                        example['%sPrevRes%d' % (home_or_away, i)] = team.last_k_matches[-i]['res']
                    else:
                        example['%sPrevRes%d' % (home_or_away, i)] = np.nan

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
        self.last_k_matches = []

    def update(self, match, home_or_away):
        self.played_matches += 1
        if match['FTR'] == home_or_away[0]:
            self.points += 3
            match['res'] = 'W'  # win
        elif match['FTR'] == 'D':
            self.points += 1
            match['res'] = 'D'
        else:
            match['res'] = 'L'  # loose
        self.scored_goals += match['FT%sG' % home_or_away[0]]
        self.conceded_goals += match['FT%sG' % ('A' if home_or_away == 'Home' else 'H')]
        self.goal_difference = self.scored_goals - self.conceded_goals
        if use_last_k_matches is not None:
            self.last_k_matches.append(match)
            self.last_k_matches = self.last_k_matches[-use_last_k_matches:]


league = League(country, division, seasons)
league.run()

def dataset_preprocessing(dataset):
    dataset = dataset.dropna()  # drop matches with Nan value in the feature
    Y = dataset['result']
    X = dataset.drop('result', axis='columns')

    # Transform categorical labels (H, D, A) into numerical values
    le = preprocessing.LabelEncoder()
    le.fit(Y)
    Y = le.transform(Y)

    # print(X.info())
    preprocessor = compose.ColumnTransformer(transformers=[
        ('cat', preprocessing.OneHotEncoder(), compose.make_column_selector(dtype_include="object")),
        ('num', preprocessing.StandardScaler(), compose.make_column_selector(dtype_exclude="object"))
    ])
    X = preprocessor.fit_transform(X)

    return X, Y

training_dataset = []
for i in range(len(seasons) - 1):
    training_dataset.append(league.datasets[seasons[i]])
training_dataset = pd.concat(training_dataset)
test_dataset = league.datasets[seasons[-1]]

# Training
logreg = LogisticRegression(C=1e5)
X, Y = dataset_preprocessing(training_dataset)
logreg.fit(X, Y)
Y_pred = logreg.predict(X)
print('Training accuracy: %.2f' % accuracy_score(Y, Y_pred))

# Test
X, Y = dataset_preprocessing(test_dataset)
Y_pred = logreg.predict(X)
print('Test accuracy: %.2f' % accuracy_score(Y, Y_pred))

# TODO: Comparison to baselines: predicting win of the home team or the best ranked
