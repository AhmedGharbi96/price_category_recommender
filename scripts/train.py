import logging

from components.training.config import TrainingConfig
from components.training.trainer import Trainer
from components.utils.get_configs import load_training_config


def run_model_training(config: TrainingConfig):
    logging.info("**** Starting training workflow ****")
    trainer = Trainer(config)
    logging.info("Loading clean data...")
    x_train, y_train, x_val, y_val = trainer.load_train_val_data()
    logging.info("Starting the training...")
    model = trainer.train_model(x_train, y_train, x_val, y_val)
    logging.info("Training the model finished.")
    logging.info("Saving the model...")
    trainer.save_model(model)
    logging.info("Model saved successfully...")
    logging.info("**** Training workflow finished successfully ****")


if __name__ == "__main__":
    training_config = load_training_config()
    run_model_training(training_config)
