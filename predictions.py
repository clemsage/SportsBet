from sklearn import preprocessing, compose
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import sklearn
import argparse
from typing import Union, Tuple

import pandas as pd
import json
import os
import warnings

import game


def dataset_preprocessing(
        dataset: pd.DataFrame,
        label_encoder: Union[preprocessing.LabelEncoder, None] = None,
        feature_preprocessor: Union[None, sklearn.base.TransformerMixin] = None) -> Tuple[pd.DataFrame,
                                                                                          pd.DataFrame,
                                                                                          preprocessing.LabelEncoder,
                                                                                          sklearn.base.TransformerMixin]:
    """
    :param dataset: Input DataFrame containing match features and the result label.
    :param label_encoder: Optional label encoder for converting categorical labels (e.g., 'H', 'D', 'A') into integers.
    If None, a new LabelEncoder is created and fit to the dataset.
    :param feature_preprocessor: Optional preprocessor for transforming features (e.g., scaling numerical features and
    encoding categorical ones). If None, a new ColumnTransformer is created and fit.
    :return: Transformed features (X) and labels (Y), along with the fitted label encoder and feature preprocessor.

    Preprocesses the dataset by encoding labels and transforming features.
    """
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


class HomeTeamWins:
    """
    Heuristics always predicting that the home team will be the winner of a match. Baseline for the ML classifier.
    """
    def __init__(self):
        self.label_encoder = None

    def infer(self, dataset: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        :param dataset: Match features
        :param kwargs: Optional keyword arguments
        :return: The predicted outcome for each input match. Both their human readable version, i.e.
        'H' (Home team wins), 'A' (Away team wins), 'D' (Draw), and their numerical version, e.g. 0 for 'H', are
        returned.
        """
        Y_pred = pd.Series(['H'] * len(dataset), index=dataset.index)
        encoded_pred = pd.Series(self.label_encoder.transform(Y_pred), index=dataset.index)
        return pd.concat([Y_pred, encoded_pred], axis=1, keys=['result', 'encoded_result'])


class BestRankedTeamWins:
    """
    Heuristics always predicting that the best ranked team will be the winner of a match. Baseline for the ML
    classifier.
    """
    def __init__(self):
        self.label_encoder = None

    def infer(self, dataset: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        :param dataset: Match features
        :param kwargs: Optional keyword arguments
        :return: The predicted outcome for each input match. Both their human readable version, i.e.
        'H' (Home team wins), 'A' (Away team wins), 'D' (Draw), and their numerical version, e.g. 0 for 'H', are
        returned.
        """
        Y_pred = pd.Series(['A'] * len(dataset), index=dataset.index)
        home_team_has_best_ranking = dataset['HomeRanking'] < dataset['AwayRanking']
        Y_pred[home_team_has_best_ranking] = 'H'
        encoded_pred = pd.Series(self.label_encoder.transform(Y_pred), index=dataset.index)
        return pd.concat([Y_pred, encoded_pred], axis=1, keys=['result', 'encoded_result'])


class ResultsPredictor(object):
    def __init__(self, league: 'game.League', **kwargs):
        """
        :param league: League to predict results for
        :param kwargs: Parsed main file arguments
        """
        # TODO: Model selection and hyperparameter tuning
        self.model_name = kwargs['model_name']
        if kwargs['config_name'] is None:
            kwargs['config_name'] = os.path.join('configs', '%s.json' % self.model_name)
        with open(kwargs['config_name'], 'r') as f:
            config = json.load(f)
        self.model = eval(self.model_name)(**config)
        print('\nModel for predicting the game results: %s' % self.model.__class__.__name__)
        self.league = league
        self.training_dataset, self.test_dataset = self.split_train_test_sets()
        self.label_encoder = None
        self.feature_preprocessor = None
        self.baselines = [HomeTeamWins(), BestRankedTeamWins()]

    def split_train_test_sets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        :return: The training dataset (N-1 first seasons) and the test dataset (last season)
        """
        training_dataset = []
        for i in range(max(1, len(self.league.seasons) - 1)):
            training_dataset.append(self.league.datasets[self.league.seasons[i].name])
        training_dataset = pd.concat(training_dataset)
        test_dataset = self.league.datasets[self.league.seasons[-1].name]
        if len(self.league.datasets) == 1:
            warnings.warn('The test season is also the training season as only one season is provided')
        return training_dataset, test_dataset

    def train(self):
        """
        :return:

        Train the ML classifier on the first N-1 seasons.
        """
        X, Y, self.label_encoder, self.feature_preprocessor = dataset_preprocessing(self.training_dataset)
        print('Feature set: %s' % list(self.training_dataset.drop('result', axis='columns').columns))
        print('Feature set size: %d' % X.shape[1])
        print('\nTraining of the model on %d samples from %s to %s seasons...' %
              (len(self.training_dataset), self.league.seasons[0].name,
               self.league.seasons[max(-len(self.league.seasons), -2)].name))
        self.model.fit(X, Y)
        Y_pred = self.model.predict(X)
        print('Training accuracy of the model: %.3f' % accuracy_score(Y, Y_pred))

    def eval(self, with_heuristics: bool = True):
        """
        :param with_heuristics: Whether to compare the ML classifier with baseline heuristics
        :return:

        Evaluate the ML classifier on the last provided season
        """
        print('\nEvaluation of the model on %d samples from the season %s...' % (
            len(self.test_dataset), self.league.seasons[-1].name))
        X, Y, _, _ = dataset_preprocessing(self.test_dataset, self.label_encoder, self.feature_preprocessor)
        Y_pred = self.model.predict(X)
        print('Test accuracy of the model: %.3f' % accuracy_score(Y, Y_pred))

        if with_heuristics:
            for baseline in self.baselines:
                baseline.label_encoder = self.label_encoder
                Y_pred = baseline.infer(self.test_dataset)
                print('Test accuracy of the %s heuristic: %.3f' % (baseline.__class__.__name__,
                                                                   accuracy_score(Y, Y_pred['encoded_result'])))

    def infer(self, dataset: pd.DataFrame, with_proba: bool = False) -> pd.DataFrame:
        """
        :param dataset: Match features
        :param with_proba: Whether to return the probabilities of model predictions
        :return: The predicted outcome for each match of the dataset
        """
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
