import logging

from components.data_processing.config import DataProcessingConfig
from components.data_processing.data_loader import DataLoader
from components.utils.get_configs import load_data_processing_config


def run_data_processing(config: DataProcessingConfig):
    logging.info("**** Starting data processing workflow ****")
    logging.info("Loading raw data...")
    loader = DataLoader(data_processing_cfg=config)
    raw_data = loader.load_raw_data(config.raw_data_file_name)
    logging.info("Splitting into train, test and validation...")
    train, test, val = loader.split_train_test_val(raw_data_df=raw_data)
    logging.info("Creating density feature...")
    train, test, val = loader.add_density_feature(train, test, val)
    logging.info("Splitting features and targets...")
    x_train, y_train = loader.split_to_feat_and_target(train)
    x_test, y_test = loader.split_to_feat_and_target(test)
    x_val, y_val = None, None
    if config.use_validation_set:
        x_val, y_val = loader.split_to_feat_and_target(val)
    if config.smote_config.use_smote:
        logging.info("Oversampling using SMOTE-NC...")
        x_train, y_train = loader.augment_training_data(x_train, y_train)
    logging.info("Encoding categorical features")
    x_train, x_test, x_val = loader.encode_categorical_feats(x_train, x_test, x_val)
    logging.info("Saving clean data...")
    loader.save_clean_dataframe(x_train, "x_train.csv")
    loader.save_clean_dataframe(y_train, "y_train.csv")

    loader.save_clean_dataframe(x_test, "x_test.csv")
    loader.save_clean_dataframe(y_test, "y_test.csv")

    if config.use_validation_set:
        loader.save_clean_dataframe(x_val, "x_val.csv")
        loader.save_clean_dataframe(y_val, "y_val.csv")
    logging.info("**** Data processing workflow finished successfully ****")


if __name__ == "__main__":
    config = load_data_processing_config()
    run_data_processing(config)
