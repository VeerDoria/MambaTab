"""
Data preprocessing pipeline.

Implements the preprocessing steps from the MambaTab paper:
1. Ordinal encode categorical/binary features
2. Leave numerical features unchanged
3. Impute missing values with mode
4. Min-max scale all features to [0, 1]
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class TabularPreprocessor:
    """Preprocessor for tabular data following MambaTab's pipeline.

    This preprocessor is designed to be fitted on training data and applied
    consistently to validation and test data, ensuring no data leakage.

    Attributes:
        label_encoders: Dictionary mapping column names to fitted LabelEncoders.
        scaler: Fitted MinMaxScaler.
        feature_columns: List of feature column names (excludes target).
        target_column: Name of the target column.
        categorical_columns: List of categorical column names.
        numerical_columns: List of numerical column names.
        modes: Dictionary mapping column names to their mode values.
    """

    def __init__(self, target_column: str = "class"):
        """Initialise the preprocessor.

        Args:
            target_column: Name of the target column.
        """
        self.target_column = target_column
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[MinMaxScaler] = None
        self.feature_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.numerical_columns: List[str] = []
        self.modes: Dict[str, any] = {}
        self._is_fitted = False

    def _identify_column_types(self, df: pd.DataFrame) -> None:
        """Identify categorical and numerical columns."""
        self.feature_columns = [
            c for c in df.columns if c != self.target_column
        ]
        self.categorical_columns = []
        self.numerical_columns = []

        for col in self.feature_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                self.categorical_columns.append(col)
            elif df[col].nunique() <= 2 and df[col].dtype in [
                "int64",
                "int32",
                "bool",
                "float64",
                "float32",
            ]:
                # Treat binary features as categorical (following paper:
                # "treat both binary and categorical features as categorical")
                self.categorical_columns.append(col)
            else:
                self.numerical_columns.append(col)

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        """Fit the preprocessor on training data.

        Args:
            df: Training DataFrame with features and target.

        Returns:
            Self (for method chaining).
        """
        self._identify_column_types(df)

        # Compute mode for each column (for imputation)
        for col in self.feature_columns:
            mode_val = df[col].mode()
            self.modes[col] = mode_val.iloc[0] if len(mode_val) > 0 else 0

        # Fit ordinal (label) encoders for categorical columns
        for col in self.categorical_columns:
            le = LabelEncoder()
            # Fill NaN before fitting
            col_data = df[col].fillna(self.modes[col]).astype(str)
            le.fit(col_data)
            self.label_encoders[col] = le

        # Fit min-max scaler on the transformed training data
        X_transformed = self._transform_features(df, fit_scaler=True)
        self._is_fitted = True

        return self

    def _transform_features(
        self, df: pd.DataFrame, fit_scaler: bool = False
    ) -> np.ndarray:
        """Transform features (encode + impute), optionally fitting the scaler.

        Args:
            df: DataFrame with feature columns.
            fit_scaler: If True, fit the MinMaxScaler on this data.

        Returns:
            Numpy array of transformed features.
        """
        result = pd.DataFrame(index=df.index)

        # Encode categorical columns
        for col in self.categorical_columns:
            col_data = df[col].fillna(self.modes.get(col, "missing")).astype(str)
            le = self.label_encoders[col]
            # Handle unseen categories gracefully
            known = set(le.classes_)
            col_data = col_data.apply(
                lambda x: x if x in known else le.classes_[0]
            )
            result[col] = le.transform(col_data)

        # Copy numerical columns (impute missing with mode)
        for col in self.numerical_columns:
            result[col] = (
                df[col]
                .fillna(self.modes.get(col, 0))
                .astype(float)
            )

        # Ensure column order matches feature_columns
        result = result[self.feature_columns]

        X = result.values.astype(np.float32)

        # Min-max scaling to [0, 1]
        if fit_scaler:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            X = self.scaler.fit_transform(X)
        else:
            if self.scaler is not None:
                X = self.scaler.transform(X)

        return X

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Transform a DataFrame into (X, y) arrays.

        Args:
            df: DataFrame with features and target.

        Returns:
            Tuple of (X, y) where X is scaled features and y is binary target.
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")

        X = self._transform_features(df, fit_scaler=False)
        y = df[self.target_column].values.astype(np.float32)

        return X, y

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fit on the data and transform it.

        Args:
            df: Training DataFrame.

        Returns:
            Tuple of (X, y).
        """
        self.fit(df)
        X = self._transform_features(df, fit_scaler=False)
        y = df[self.target_column].values.astype(np.float32)
        return X, y

    @property
    def num_features(self) -> int:
        """Number of input features after preprocessing."""
        return len(self.feature_columns)
