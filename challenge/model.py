import pickle

import numpy as np
import pandas as pd
import xgboost as xgb

from datetime import datetime
from typing import Tuple, Union, List

from sklearn.model_selection import train_test_split


THRESHOLD_IN_MINUTES = 15
MODEL_FILE_NAME = "delay_model.pkl"


class DelayModel:

    def __init__(
        self
    ):
        self._features = [
            "OPERA_Latin American Wings",
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]
        self._model = self.__load_model(MODEL_FILE_NAME)

    def __load_model(self, file_name):
        try:
            with open(file_name, 'rb') as fp:
                return pickle.load(fp)
        except FileNotFoundError:
            return None
        
    def __save_model(self, filename):
        with open(filename, 'wb') as fp:
            pickle.dump(self._model, fp)

    def get_min_diff(self, data):
        """
        Calculate the minute difference between two datetime values.

        Args:
            data (pd.DataFrame): raw data.

        Returns:
            float: Minute difference between 'Fecha-O' and 'Fecha-I'.
        """

        fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
        return min_diff

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """

        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix='MES')],
            axis=1
        )

        for feature in self._features:
            if feature not in features.columns:
                features[feature] = 0

        if target_column:
            data['min_diff'] = data.apply(self.get_min_diff, axis=1)

            data[target_column] = np.where(
                data['min_diff'] > THRESHOLD_IN_MINUTES, 1, 0
            )

            return features[self._features], data[[target_column]]
        else:
            return features[self._features]

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """

        x_train, _, y_train, _ = train_test_split(
            features, target, test_size=0.33, random_state=42
        )

        n_y0 = int((target == 0).sum())
        n_y1 = int((target == 1).sum())
        scale = n_y0 / n_y1

        self._model = xgb.XGBClassifier(
            random_state=1, learning_rate=0.01, scale_pos_weight=scale
        )

        self._model.fit(x_train, y_train)
        self.__save_model(MODEL_FILE_NAME)

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """

        if self._model is None:
            self.__load_model(MODEL_FILE_NAME)

        predictions = self._model.predict(features)

        return predictions.tolist()
