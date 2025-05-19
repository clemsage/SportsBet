import urllib
import os
from pathlib import Path
from typing import List, Union, Dict
from collections import defaultdict
import datetime
from copy import deepcopy

import pandas as pd
import numpy as np

import betting

#################
base_path_data = 'https://www.football-data.co.uk/mmz4281'
FTR2Points = {'H': 3, 'D': 1, 'A': 0}  # Number of points won by the home team of a match ('H' for 'Home', 'D' for
# 'Draw', 'A' for 'Away')
#################


def get_season_ids(
        start_season: str,
        end_season: str,
        offset: Union[List[int], None] = None) -> List[str]:
    """
    :param start_season: Two digit year indicating the first season analyzed, e.g. 04 for the 2004/2005 season.
    :param end_season: Two digit year indicating the last season analyzed, e.g. 05 for the 2004/2005 season.
    :param offset: Number of seasons to offset for both start and end seasons, e.g. if [-1, 1] is provided along the
     above arguments examples, then the seasons 2003/2004 (one season sooner) up to 2005/2006 will be considered. If
     not provided, no offset is applied.
    :return: The IDs of all the league seasons between *start_season* and *end_season*
    """
    start_season, end_season = int(start_season), int(end_season)
    if start_season > end_season:  # Starting (resp. end) season is in the XXe (resp. XXIe) century
        end_season += 100

    if offset is None:
        offset = [0, 0]

    seasons = []
    for year in range(start_season + offset[0], end_season + offset[1]):
        seasons.append('%s%s' % (str(year % 100).zfill(2), str((year + 1) % 100).zfill(2)))
    return seasons


class League(object):
    def __init__(self, betting_platforms: List[str], **kwargs):
        """
        :param betting_platforms: List of betting platforms tickers, e.g. 'BW' for Bet&Win platform.
        :param kwargs: Parsed main file arguments
        """
        self.country = kwargs['country']
        print('Country: %s' % self.country)
        self.division = kwargs['division']
        print('Division: %s' % self.division)
        id_country = self.country[0].upper()
        if self.country.lower() == 'spain':
            id_country = 'SP'
        elif self.country.lower() == 'germany':
            id_country = 'D'
        elif self.country.lower() == 'england':
            self.division -= 1  # to follow the id of the website from which we pull the results
        self.name = id_country + str(self.division)
        match_historic = [] if kwargs['number_previous_direct_confrontations'] else None

        seasons = get_season_ids(kwargs['start_season'], kwargs['end_season'])
        print("Analyzing the seasons from %s to %s..." % (seasons[0], seasons[-1]))
        self.seasons = [Season(self.name, season, match_historic, betting_platforms, **kwargs) for season in seasons]
        assert len(self.seasons) >= 1, "We have not found any season for start_season=%s and end_season=%s" % \
                                       (kwargs['start_season'], kwargs['end_season'])
        self.datasets = {}
        self.betting_platforms = betting_platforms

    def run(self):
        """
        :return:

        Run the matches for all seasons to gather a dataset for the ML model training and testing
        """
        for season in self.seasons:
            season.run()
            self.datasets[season.name] = season.dataset

    def analyze_betting_platforms_margins(self):
        """
        :return:

        Analyze the average margins of betting platforms by summing the inverse of their home, away and draw odds.
        """
        margins = {}
        all_matches = pd.concat([season.matches for season in self.seasons])
        output = 'Average margin of each betting platform per match:'
        for platform in self.betting_platforms:
            odd_tickers = {platform + result for result in ['H', 'D', 'A']}
            if len(odd_tickers.intersection(all_matches)) == 3:
                odds = all_matches.loc[:, list(odd_tickers)].dropna()
                inv_odds = 1.0 / odds
                probs = inv_odds.sum(axis=1)
                margins[platform] = probs.mean()
                output = output + ' %s: %.1f%%,' % (platform, 100*margins[platform]-100)
            else:
                margins[platform] = np.nan
        margins['average'] = np.nanmean(list(margins.values()))
        print(output + ' average: %.1f%%' % (100*margins['average']-100))


class Season(object):
    def __init__(
            self,
            league_name: str,
            name: str,
            match_historic: Union[List, None],
            betting_platforms: List[str],
            **kwargs):
        """
        :param league_name: Name of the league, e.g. 'SP1' for 1st Spanish division
        :param name: Four digits ID of the season, e.g. '0405' for the 2004/2005 season
        :param match_historic: List of matches from previous seasons that were already loaded. None can also be passed
        if the previous matches are not needed.
        :param betting_platforms: List of betting platforms tickers, e.g. 'BW' for Bet&Win platform
        :param kwargs: Parsed main file arguments
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.league_name = league_name
        self.name = name
        self.betting_platforms = betting_platforms
        self.matches = self.get_season_matches(name, league_name)
        if match_historic is not None:
            if not len(match_historic):
                previous_seasons = get_season_ids(
                    start_season=self.name[:2], end_season=self.name[2:],
                    offset=[-self.number_previous_direct_confrontations - 1, -1])
                for season in previous_seasons:
                    match_historic.append(self.get_season_matches(season, league_name))
            self.match_historic = pd.concat(match_historic)
            match_historic.append(self.matches)
        self._matches = None
        self.teams = None
        self.ranking = None
        self.dataset = None
        self.clear_data()

    @staticmethod
    def get_season_matches(name: str, league_name: str) -> pd.DataFrame:
        """
        :param name: Four digits ID of the season, e.g. '0405' for the 2004/2005 season
        :param league_name: Name of the league, e.g. 'SP1' for 1st Spanish division
        :return: The season's matches data as available on the football-data.co.uk website.

        Load the match results for the given season and league. The matches are also locally saved for faster/offline
        loading during the next script executions.
        """
        season_id = '/'.join((name, league_name + '.csv'))
        local_path = '/'.join(('data', season_id))
        if os.path.exists(local_path):  # Load matches from local file
            matches = pd.read_csv(local_path, sep=',')
        else:  # Load matches from football-data.co.uk website
            data_url = '/'.join((base_path_data, season_id))
            try:
                matches = pd.read_csv(data_url, sep=',')
            except urllib.error.HTTPError:
                print('The following data URL seems incorrect: %s' % data_url)
                raise Exception('Check the URL')
            except pd.errors.ParserError as err:  # extra empty columns are provided for some rows, just ignore them
                print(err)
                columns = pd.read_csv(data_url, sep=',', nrows=1).columns.tolist()
                matches = pd.read_csv(data_url, sep=',', names=columns, skiprows=1)

            Path(os.path.split(local_path)[0]).mkdir(parents=True, exist_ok=True)
            matches.to_csv(local_path, index=False)
        matches = matches.dropna(how='all')

        def normalize_year(year: str) -> str:
            """
            :param year: Two or four digit long year
            :return: Four digit long year

            Normalize the year for 2017/2018 French 1st league since the file names on the football-data.co.uk website
             follow the DD/MM/YY format instead of the DD/MM/YYYY format used for other leagues
            """
            if len(year) == 2:
                current_year = int(str(datetime.datetime.now().year)[-2:])
                if int(year) <= current_year:
                    year = '20' + year  # XXIe century
                else:
                    year = '19' + year  # XXe century
            return year

        # Sort the matches by chronological order
        matches['day'] = matches['Date'].apply(lambda x: x.split('/')[0])
        matches['month'] = matches['Date'].apply(lambda x: x.split('/')[1])
        matches['year'] = matches['Date'].apply(lambda x: normalize_year(x.split('/')[2]))
        matches['Date'] = matches.apply(lambda df: '/'.join((df['day'], df['month'], df['year'])), axis=1)
        matches['Date'] = pd.to_datetime(matches['Date'], format='%d/%m/%Y')
        matches.sort_values(by=['Date'], inplace=True)

        return matches

    def clear_data(self):
        """
        :return:

        Clear the season data
        """
        team_names = self.matches['HomeTeam'].unique()
        self.teams = {team_name: Team(team_name, self.match_history_length) for team_name in team_names}
        self.ranking = self.get_ranking()
        self.dataset = []

    def update_statistics(self, played_matches: pd.DataFrame):
        """
        :param played_matches: Matches that were played at the current date
        :return:
        """
        for stat in ['FTR', 'FTHG', 'FTAG']:
            assert stat in played_matches, '%s statistics must be available' % stat

        for _, match in played_matches.iterrows():
            assert match['FTR'] in ['H', 'D', 'A'], '%s is an unknown match result' % match['FTR']
            for home_or_away in ['Home', 'Away']:
                self.teams[match['%sTeam' % home_or_away]].update(match, home_or_away)
        self.ranking = self.get_ranking()

    def get_ranking(self) -> pd.DataFrame:
        """
        :return: The ranking of teams for the current date.
        """
        ranking_props = ['name', 'played_matches', 'points', 'goal_difference', 'scored_goals', 'conceded_goals']
        ranking = pd.DataFrame([{key: value for key, value in vars(team).items() if key in ranking_props}
                                for team in self.teams.values()])
        ranking.set_index('name', inplace=True)
        ranking.sort_values(['points', 'goal_difference'], ascending=False, inplace=True)
        for team in self.teams.values():
            team.ranking = 1 + ranking.index.get_loc(team.name)
        return ranking

    def run(self, betting_strategy: Union[betting.BettingStrategy, None] = None):
        """
        :param betting_strategy: Optional betting strategy to apply while running the season.
        :return:

        Run the whole season matchday by matchday and prepare a dataset for ML model training and testing.
        """
        self.clear_data()
        self._matches = deepcopy(self.matches)
        if betting_strategy is not None:
            print('\nLeveraging the predictive models to bet for the %s season...' % self.name)
        while len(self._matches):
            # Group the matches by date
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

    def prepare_example(self, match: pd.Series) -> Dict:
        """
        :param match: Data of a football match
        :return:

        Gather features and label about the match for later training and evaluating a ML model to predict the outcome
        of the match (win, loose, draw)
        """
        example = {'result': match['FTR']}  # ground truth

        # Gather numerical features for both home and away teams
        for home_or_away in ['Home', 'Away']:
            team_name = match['%sTeam' % home_or_away]
            team = self.teams[team_name]

            # Current league ranking features
            example['%sPlayedMatches' % home_or_away] = team.played_matches
            example['%sRanking' % home_or_away] = team.ranking
            example['%sAvgPoints' % home_or_away] = np.divide(team.points, team.played_matches)

            # Features related to the most recent matches against other teams in the league
            if self.match_history_length is not None:
                for i in range(1, 1 + self.match_history_length):
                    key = '%sPrev%d' % (home_or_away, i)
                    if i <= len(team.last_k_matches[home_or_away]):
                        prev_match = team.last_k_matches[home_or_away][-i]
                        if prev_match['Res'] == 'D':
                            # = Odd that this team won - Odd that the other team won
                            coeffs = defaultdict(lambda: -1)
                            coeffs[home_or_away[0]] = 1
                        elif prev_match['Res'] == 'W':
                            # = Odd that this team won
                            coeffs = defaultdict(lambda: 0)
                            coeffs[home_or_away[0]] = 1
                        elif prev_match['Res'] == 'L':
                            # = - Odd that the other team won
                            coeffs = defaultdict(lambda: -1)
                            coeffs[home_or_away[0]] = 0
                        else:
                            raise Exception('A match result is either a draw (D), a win (W) or a loose (L)')

                        # Score comparing the betting odds and the actual results to gauge the team form
                        current_form_score = 0
                        for prev_home_or_away in ['H', 'A']:
                            odd_tickers = {platform + prev_home_or_away for platform in self.betting_platforms}
                            available_tickers = list(odd_tickers.intersection(prev_match.keys()))
                            odd_result = prev_match.loc[available_tickers].mean()
                            current_form_score += coeffs[prev_home_or_away] * odd_result
                        example[key] = current_form_score
                    else:
                        example[key] = np.nan

        # Features related to the direct confrontations of the home and away teams in the past seasons
        if self.number_previous_direct_confrontations:
            previous_confrontations = self.match_historic[
                (self.match_historic['HomeTeam'] == match['HomeTeam']) &
                (self.match_historic['AwayTeam'] == match['AwayTeam'])]
            previous_confrontations = previous_confrontations[-self.number_previous_direct_confrontations:]
            for i in range(1, 1 + self.number_previous_direct_confrontations):
                key = 'PrevConfrFTR%d' % i
                if i <= len(previous_confrontations):  # TODO: add also time spent since these confrontations ?
                    example[key] = previous_confrontations.iloc[-i]['FTR']
                    if self.match_results_encoding == 'points':
                        example[key] = FTR2Points[example[key]]
                else:
                    example[key] = np.nan
        return example


class Team(object):
    def __init__(self, name: str, match_history_length: Union[None, int]):
        """
        :param name: Name of the team, e.g. Man City
        :param args: Parsed main file arguments
        """
        self.name = name
        self.match_history_length = match_history_length

        # Current season attributes
        self.played_matches = 0
        self.points = 0
        self.goal_difference = 0
        self.scored_goals = 0
        self.conceded_goals = 0
        self.ranking = None
        self.last_k_matches = {'Home': [], 'Away': []}

    def update(self, match: pd.Series, home_or_away: str):
        """
        :param match: Match involving the team
        :param home_or_away: Whether the team is the 'Home' or 'Away' team for this match
        :return:

        Update the team's season attributes with the input match
        """
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
        if self.match_history_length is not None:
            self.last_k_matches[home_or_away].append(match)
            self.last_k_matches[home_or_away] = self.last_k_matches[home_or_away][-self.match_history_length:]
