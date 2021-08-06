import urllib
import os
from pathlib import Path

import datetime
from copy import deepcopy
import pandas as pd
import numpy as np
from collections import defaultdict

#################
base_path_data = 'https://www.football-data.co.uk/mmz4281'
FTR2Points = {'H': 3, 'D': 1, 'A': 0}
#################


def get_season_ids(start_season, end_season=None, offset=[0, 0]):
    if isinstance(start_season, str):
        start_season = int(start_season[:2])
    if end_season is None:
        end_season = start_season
    if start_season > end_season:  # Starting (resp. end) season is in the XXe (resp. XXIe) century
        end_season += 100
    seasons = []
    for year in range(start_season + offset[0], end_season + offset[1]):
        seasons.append('%s%s' % (str(year % 100).zfill(2), str((year + 1) % 100).zfill(2)))
    return seasons


class League(object):
    def __init__(self, args, betting_platforms):
        self.country = args.country
        print('Country: %s' % args.country)
        self.division = args.division
        print('Division: %s' % args.division)
        id_country = args.country[0].upper()
        if args.country.lower() == 'spain':
            id_country = 'SP'
        elif args.country.lower() == 'germany':
            id_country = 'D'
        elif args.country.lower() == 'england':
            self.division -= 1  # to follow the id of the website from which we pull the results
        self.name = id_country + str(self.division)
        match_historic = [] if args.number_previous_direct_confrontations else None

        seasons = get_season_ids(args.start_season, args.end_season)
        print("Analyzing the seasons from %s to %s..." % (seasons[0], seasons[-1]))
        self.seasons = [Season(self.name, season, match_historic, args, betting_platforms) for season in seasons]
        assert len(self.seasons) >= 1, "We have not found any season for start_season=%d and end_season=%d" % \
                                       (args.start_season, args.end_season)
        self.datasets = {}

    def run(self):
        for season in self.seasons:
            season.run()
            self.datasets[season.name] = season.dataset


class Season(object):
    def __init__(self, league_name, name, match_historic, args, betting_platforms):
        self.args = args
        self.league_name = league_name
        self.name = name
        self.betting_platforms = betting_platforms
        self.matches = get_season_matches(name, league_name)
        if match_historic is not None:
            if not len(match_historic):
                for season in get_season_ids(self.name, offset=[-args.number_previous_direct_confrontations - 1, -1]):
                    match_historic.append(get_season_matches(season, league_name))
            self.match_historic = pd.concat(match_historic)
            match_historic.append(self.matches)
        self._matches = None
        self.teams = None
        self.ranking = None
        self.dataset = None
        self.reset_statistics()

    def reset_statistics(self):
        team_names = self.matches['HomeTeam'].unique()
        self.teams = {team_name: Team(team_name, self.args) for team_name in team_names}
        self.ranking = self.get_ranking()
        self.dataset = []

    def update_statistics(self, played_matches):
        for stat in ['FTR', 'FTHG', 'FTAG']:
            assert stat in played_matches, '%s statistics must be available' % stat

        for _, match in played_matches.iterrows():
            assert match['FTR'] in ['H', 'D', 'A'], '%s is an unknown match result' % match['FTR']
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
        if betting_strategy is not None:
            print('\nLeveraging the predictive models to bet for the %s season...' % self.name)
        while len(self._matches):
            # Group matches by date
            current_date = self._matches['Date'].iloc[0]
            matches = self._matches.loc[self._matches['Date'] == current_date]
            dataset = []
            for _, match in matches.iterrows():
                dataset.append(self.prepare_example(match))
            dataset = pd.DataFrame(dataset, index=matches.index)
            if betting_strategy is not None:
                betting_strategy.apply(dataset, matches)
                betting_strategy.record_bankroll(current_date)
            dataset = dataset.dropna()  # drop the matches with Nan in the features, i.e. usually the first
            # game days since the season's match historic is empty
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
            example['%sPlayedMatches' % home_or_away] = team.played_matches
            example['%sRanking' % home_or_away] = team.ranking
            example['%sAvgPoints' % home_or_away] = np.divide(team.points, team.played_matches)

            if self.args.match_history_length is not None:
                for i in range(1, 1 + self.args.match_history_length):
                    key = '%sPrev%d' % (home_or_away, i)
                    if i <= len(team.last_k_matches[home_or_away]):
                        prev_match = team.last_k_matches[home_or_away][-i]
                        if prev_match['Res'] == 'D':
                            coeffs = defaultdict(lambda: -1)
                            coeffs[home_or_away[0]] = 1
                        elif prev_match['Res'] == 'W':
                            coeffs = defaultdict(lambda: 0)
                            coeffs[home_or_away[0]] = 1
                        elif prev_match['Res'] == 'L':
                            coeffs = defaultdict(lambda: -1)
                            coeffs[home_or_away[0]] = 0
                        else:
                            raise Exception('A match result is either a draw (D), a win (W) or a loose (L)')
                        current_form_score = 0
                        for prev_home_or_away in ['H', 'A']:
                            odd_tickers = {platform + prev_home_or_away for platform in self.betting_platforms}
                            available_tickers = odd_tickers.intersection(prev_match.keys())
                            odd_result = prev_match.loc[available_tickers].mean()
                            current_form_score += coeffs[prev_home_or_away] * odd_result
                        example[key] = current_form_score
                    else:
                        example[key] = np.nan

        if self.args.number_previous_direct_confrontations:
            previous_confrontations = self.match_historic[
                (self.match_historic['HomeTeam'] == match['HomeTeam']) &
                (self.match_historic['AwayTeam'] == match['AwayTeam'])]
            previous_confrontations = previous_confrontations[-self.args.number_previous_direct_confrontations:]
            for i in range(1, 1 + self.args.number_previous_direct_confrontations):
                key = 'PrevConfrFTR%d' % i
                if i <= len(previous_confrontations):  # TODO: add also time spent since these confrontations ?
                    example[key] = previous_confrontations.iloc[-i]['FTR']
                    if self.args.match_results_encoding == 'points':
                        example[key] = FTR2Points[example[key]]
                else:
                    example[key] = np.nan
        return example


class Team(object):
    def __init__(self, name, args):
        self.name = name
        self.played_matches = 0
        self.points = 0
        self.goal_difference = 0
        self.scored_goals = 0
        self.conceded_goals = 0
        self.ranking = None
        self.last_k_matches = {'Home': [], 'Away': []}
        self.args = args

    def update(self, match, home_or_away):
        match = match.copy()
        self.played_matches += 1
        if match['FTR'] == home_or_away[0]:
            points = 3
            match['Res'] = 'W'  # win
        elif match['FTR'] == 'D':
            points = 1
            match['Res'] = 'D'
        else:
            points = 0
            match['Res'] = 'L'  # loose
        self.points += points
        match['Points'] = points
        self.scored_goals += match['FT%sG' % home_or_away[0]]
        self.conceded_goals += match['FT%sG' % ('A' if home_or_away == 'Home' else 'H')]
        self.goal_difference = self.scored_goals - self.conceded_goals
        if self.args.match_history_length is not None:
            self.last_k_matches[home_or_away].append(match)
            self.last_k_matches[home_or_away] = self.last_k_matches[home_or_away][-self.args.match_history_length:]


def get_season_matches(name, league_name):
    season_id = '/'.join((name, league_name + '.csv'))
    local_path = '/'.join(('data', season_id))
    if os.path.exists(local_path):
        matches = pd.read_csv(local_path, sep=',', encoding='mbcs')
    else:
        data_url = '/'.join((base_path_data, season_id))
        try:
            matches = pd.read_csv(data_url, sep=',', encoding='mbcs')
        except urllib.error.HTTPError:
            print('The following data URL seems incorrect: %s' % data_url)
            raise Exception('Check the URL')
        except pd.errors.ParserError as err:  # extra empty columns are provided for some rows, just ignore them
            print(err)
            columns = pd.read_csv(data_url, sep=',', nrows=1).columns.tolist()
            matches = pd.read_csv(data_url, sep=',', encoding='mbcs', names=columns, skiprows=1)

        Path(os.path.split(local_path)[0]).mkdir(parents=True, exist_ok=True)
        matches.to_csv(local_path, index=False)
    matches = matches.dropna(how='all')

    # sort matches by chronological order
    def normalize_year(year):  # fix for 2017/2018 French 1st league having DD/MM/YY format instead of DD/MM/YYYY
        if len(year) == 2:
            current_year = int(str(datetime.datetime.now().year)[-2:])
            if int(year) <= current_year:
                year = '20' + year
            else:
                year = '19' + year
        return year
    matches['day'] = matches['Date'].apply(lambda x: x.split('/')[0])
    matches['month'] = matches['Date'].apply(lambda x: x.split('/')[1])
    matches['year'] = matches['Date'].apply(lambda x: normalize_year(x.split('/')[2]))
    matches['Date'] = matches.apply(lambda df: '/'.join((df['day'], df['month'], df['year'])), axis=1)
    matches['Date'] = pd.to_datetime(matches['Date'], format='%d/%m/%Y')
    matches.sort_values(by=['Date'], inplace=True)

    return matches
