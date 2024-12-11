import pandas as pd
from sklearn.preprocessing import LabelEncoder


class CategoricalEncoder:
    def __init__(self, categorical_columns: list[str]) -> None:
        self.categorical_columns = categorical_columns
        self.encoders = {}

    def fit(self, df: pd.DataFrame) -> None:
        for col in self.categorical_columns:
            self.encoders[col] = LabelEncoder().fit(df[col])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.categorical_columns:
            df[col] = self.encoders[col].transform(df[col])
        return df

    def decode(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.categorical_columns:
            print(f"Decoding column {col}")
            df[col] = self.encoders[col].inverse_transform(df[col])
        # remove line when column acctsessionid is nan
        df = df.dropna(subset=["acctsessionid"])
        return df


class PostProcess:
    def __init__(
        self, categorical_columns: list[str], original_data: pd.DataFrame
    ) -> None:
        self.categorical_columns = categorical_columns
        self.original_data = original_data
        self.col_types = self._classify_column_types(original_data)

    # Function to classify column types
    def _classify_column_types(self, df):
        column_types = {}
        for col, dtype in df.dtypes.items():
            if pd.api.types.is_integer_dtype(dtype):
                column_types[col] = "integer"
            elif pd.api.types.is_float_dtype(dtype):
                column_types[col] = "float"
            elif pd.api.types.is_object_dtype(
                dtype
            ) or pd.api.types.is_categorical_dtype(dtype):
                column_types[col] = "categorical"
            else:
                column_types[col] = "other"

        return column_types

    @staticmethod
    def min_max_rescale(value, min_value, max_value):
        if value < min_value:
            return min_value
        elif value > max_value:
            return max_value
        else:
            return value

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        min_values = self.original_data.min()
        max_values = self.original_data.max()

        for feature in df.columns:
            df[feature] = df[feature].apply(
                lambda x: PostProcess.min_max_rescale(
                    x, min_values[feature], max_values[feature]
                )
            )

        # Convert categorical columns using LabelEncoder and apply rounding
        for col in self.categorical_columns:
            df[col] = df[col].round()
            df[col] = df[col].map(lambda x: 0 if x == -0 else x).astype(int)

        for col in df.columns:
            if self.col_types[col] == "integer":
                df[col] = df[col].astype(int)
            elif self.col_types[col] == "float":
                df[col] = df[col].astype(float)
            elif self.col_types[col] == "categorical":
                df[col] = df[col].astype(int)

        return df


# standardize numerical features
