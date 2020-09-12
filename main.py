"""
Main file for the Sports Bet project
"""
import pandas as pd
pd.options.display.width = 0

#################
# DO NOT CHANGE #
base_path_data = 'https://www.football-data.co.uk/mmz4281'
#################

country = 'France'
division = '1'
seasons = ['1819']  # list of 4 digits, e.g. 1819 for the 2018/2019 season

###########################
# Features for prediction #
use_last_k_matches = 5  # None to not use it or integer (negative value for taking into account all previous matches)
###########################


class League(object):
    def __init__(self, country, division, seasons):
        self.country = country
        self.division = division
        self.name = country[0].upper() + division
        self.seasons = [Season(self.name, season) for season in seasons]


class Season(object):
    def __init__(self, league_name, name):
        self.league_name = league_name
        self.name = name
        data_url = '/'.join((base_path_data, name, league_name + '.csv'))
        self.matches = pd.read_csv(data_url, sep=',', encoding='mbcs')
        self.matches['Date'] = pd.to_datetime(self.matches['Date'], format='%d/%m/%Y')
        self.matches.sort_values(by=['Date'], inplace=True)
        team_names = self.matches['HomeTeam'].unique()
        self.teams = {team_name: Team(team_name) for team_name in team_names}
        self.ranking = self.get_ranking()

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
        self.update_statistics(self.matches)


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
        elif match['FTR'] == 'D':
            self.points += 1
        self.scored_goals += match['FT%sG' % home_or_away[0]]
        self.conceded_goals += match['FT%sG' % ('A' if home_or_away == 'Home' else 'H')]
        self.goal_difference = self.scored_goals - self.conceded_goals
        if use_last_k_matches is not None:
            self.last_k_matches.append(match)
            if use_last_k_matches > 0:
                self.last_k_matches = self.last_k_matches[-use_last_k_matches:]


league = League(country, division, seasons)
league.seasons[0].run()
print(league.seasons[0].ranking)
