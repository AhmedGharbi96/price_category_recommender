import multiprocessing
import pickle
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.nonparametric.kernel_density as statmKDE
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split

from components.data_processing.config import DataProcessingConfig
from components.utils.constants import DATA_FOLDER_PATH


class DataLoader:
    def __init__(self, data_processing_cfg: DataProcessingConfig) -> None:
        self.config = data_processing_cfg
        self.clean_data_path = Path(
            DATA_FOLDER_PATH / "clean" / self.config.experiment_id
        )
        self.clean_data_path.mkdir(parents=True, exist_ok=True)

    def save_clean_dataframe(self, df: pd.DataFrame, file_name: str) -> None:
        """Given a clean dataframe, save the dataframe under data/clean

        Args:
            df (pd.DataFrame): dataframe to save
            file_name (str): name of the file
        """

        df.to_csv(self.clean_data_path / file_name, index=False)

    def load_raw_data(self, file_name: str) -> pd.DataFrame:
        """Load csv raw data file from data/raw

        Args:
            file_name (str): file name to load

        Returns:
            pd.DataFrame: dataframe to return
        """
        raw_data_path = Path(DATA_FOLDER_PATH / "raw")
        assert raw_data_path.exists(), "'raw' folder does not exist under 'data'"
        df = pd.read_csv(raw_data_path / file_name, index_col=[0])
        df["floorsTotal"] = df["floorsTotal"].astype(int)
        return df

    def split_train_test_val(
        self, raw_data_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        assert raw_data_df.shape[0] > self.config.test_set_n_samples
        # last test_set_n_samples are for the test set
        train_val_data = raw_data_df.iloc[: -self.config.test_set_n_samples]
        test_data = raw_data_df.iloc[-self.config.test_set_n_samples :].copy()
        if self.config.use_validation_set:
            train_data, val_data = train_test_split(
                train_val_data,
                test_size=self.config.validation_set_ratio,
                stratify=train_val_data["price_category"],
                random_state=self.config.random_state,
            )
        else:
            train_data, val_data = train_val_data, None
        return train_data, test_data, val_data

    def add_density_feature(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """Train a kernel density estimator on the latitude and longitude features,
        saves the kernel for future use and add 'density' feature to the dataframes.
        The intuition behind the 'density' feature is that real estates in dense
        regions tends to be expensive, while less dense regions are less expensive.

        The implementation uses statsmodel instead of sklearn because statsmodel
        supports multi-dimensional bandwidth input but does not support 'Haversine'
        distance which should be used for our use case. The spherical distortion
        should not affect our results severely because the points are close to each
        other.

        Args:
            train_data (pd.DataFrame): train data
            test_data (pd.DataFrame): test data
            val_data (pd.DataFrame): val data

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]: dataframes
                after adding the 'density' feature.
        """

        # fit kde on train data, then use fitted estimator on test and val
        kde = statmKDE.KDEMultivariate(
            data=np.radians(train_data[["latitude", "longitude"]]),
            var_type="cc",
            bw=self.config.kde_config.bandwidth_estimate_method,
        )
        kde_folder = self.clean_data_path / "kde"
        kde_folder.mkdir(parents=True, exist_ok=True)
        with open(kde_folder / "kde.pkl", "wb") as handle:
            pickle.dump(kde, handle, protocol=pickle.HIGHEST_PROTOCOL)

        def parrallel_score_samples(
            kde, samples, thread_count=int(0.875 * multiprocessing.cpu_count())
        ):
            # utility for faster pdf calculations
            with multiprocessing.Pool(thread_count) as p:
                return np.concatenate(
                    p.map(kde.pdf, np.array_split(samples, thread_count))
                )

        pdf_train = parrallel_score_samples(
            kde, np.radians(train_data[["latitude", "longitude"]])
        )
        train_data.loc[:, "density"] = pdf_train
        # plot density for train data
        plt.figure(figsize=(12, 8))
        plt.scatter(
            x=train_data["latitude"],
            y=train_data["longitude"],
            c=pdf_train,
            alpha=0.3,
            s=10,
            marker="o",
        )
        plt.colorbar()
        plt.title("Points density - training data")
        plt.savefig(kde_folder / "train_data_density.png")
        if val_data is not None:
            pdf_val = parrallel_score_samples(
                kde, np.radians(val_data[["latitude", "longitude"]])
            )
            val_data.loc[:, "density"] = pdf_val

        pdf_test = parrallel_score_samples(
            kde, np.radians(test_data[["latitude", "longitude"]])
        )
        test_data.loc[:, "density"] = pdf_test
        return train_data, test_data, val_data

    def split_to_feat_and_target(
        self, df: pd.DataFrame, target_name: Optional[str] = "price_category"
    ):
        x = df.loc[:, df.columns != target_name]
        y = df.loc[:, target_name]
        return x, y

    def augment_training_data(
        self, x: pd.DataFrame, y: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        categorical_columns = x.select_dtypes(include=["object", "category"]).columns
        cat_indices = []
        for cat in categorical_columns:
            cat_indices.append(x.columns.get_loc(key=cat))
        categories, counts = np.unique(y, return_counts=True)
        sampling_strategy = {
            cat: self.config.smote_config.min_n_samples
            if cat_count < self.config.smote_config.min_n_samples
            else cat_count
            for cat, cat_count in zip(categories, counts)
        }
        sm = SMOTENC(
            random_state=self.config.random_state,
            categorical_features=cat_indices,
            k_neighbors=self.config.smote_config.k_neighbors,
            sampling_strategy=sampling_strategy,
        )
        x, y = sm.fit_resample(x, y)
        return x, y
