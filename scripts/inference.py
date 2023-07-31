import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from components.inference.config import InferenceConfig
from components.inference.inferrer import Inferrer
from components.inference.metrics import (
    micro_f1_score,
    micro_precision_score,
    micro_recall_score,
    soft_micro_f1_score,
    soft_micro_precision_score,
    soft_micro_recall_score,
    weighted_cohen_kappa_score,
)
from components.utils.constants import DATA_FOLDER_PATH
from components.utils.get_configs import load_inference_config


def run_inference(config: InferenceConfig):
    logging.info("**** Starting inference workflow ****")
    inference_data_path = DATA_FOLDER_PATH / "inference" / config.experiment_id
    inference_data_path.mkdir(parents=True, exist_ok=True)
    logging.info("Loading model and data...")
    model = Inferrer(config)
    x_test, y_test = model.load_inference_data()

    logging.info("Predicting test dataset...")
    # get and save predictions
    preds = model.predict_top_k(x_test, 2)
    preds_df = pd.concat(
        [
            x_test,
            pd.DataFrame({"first_range": preds[:, 0], "second_range": preds[:, 1]}),
        ],
        axis=1,
    )
    preds_df.to_csv(inference_data_path / "x_test_predictions.csv", index=False)

    logging.info("Calculating metrics...")
    # category-specific metrics
    recall_per_cat = micro_recall_score(model, x_test, y_test, per_class_score=True)
    soft_recall_per_cat = soft_micro_recall_score(
        model, x_test, y_test, per_class_score=True
    )

    precision_per_cat = micro_precision_score(
        model, x_test, y_test, per_class_score=True
    )
    soft_precision_per_cat = soft_micro_precision_score(
        model, x_test, y_test, per_class_score=True
    )

    f1_per_cat = micro_f1_score(model, x_test, y_test, per_class_score=True)
    soft_f1_per_cat = soft_micro_f1_score(model, x_test, y_test, per_class_score=True)

    index = pd.Index([f"price_category {i}" for i in np.sort(np.unique(y_test))])
    data = {
        "recall": recall_per_cat,
        "soft recall": soft_recall_per_cat,
        "precision": precision_per_cat,
        "soft precision": soft_precision_per_cat,
        "f1": f1_per_cat,
        "soft f1": soft_f1_per_cat,
    }
    cat_specific_metrics = pd.DataFrame(index=index, data=data)
    cat_specific_metrics.to_csv(inference_data_path / "category_specific_metrics.csv")

    # general performance metrics
    recall = micro_recall_score(model, x_test, y_test, per_class_score=False)
    soft_recall = soft_micro_recall_score(model, x_test, y_test, per_class_score=False)

    precision = micro_precision_score(model, x_test, y_test, per_class_score=False)
    soft_precision = soft_micro_precision_score(
        model, x_test, y_test, per_class_score=False
    )

    f1 = micro_f1_score(model, x_test, y_test, per_class_score=False)
    soft_f1 = soft_micro_f1_score(model, x_test, y_test, per_class_score=False)
    weighted_cohen_kappa = weighted_cohen_kappa_score(model, x_test, y_test)
    data = {
        "recall": recall,
        "soft recall": soft_recall,
        "precision": precision,
        "soft precision": soft_precision,
        "f1": f1,
        "soft f1": soft_f1,
        "weighted_cohen_kappa": weighted_cohen_kappa,
    }
    general_performance_metrics = pd.DataFrame(data=data, index=[0])
    general_performance_metrics.to_csv(
        inference_data_path / "general_performance_metrics.csv"
    )

    # calculate and save confusion matrix
    cm = confusion_matrix(y_test, model.predict(x_test))
    classes = [f"price_category {i}" for i in np.sort(np.unique(y_test))]
    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true=y_test,
        y_pred=model.predict(x_test),
        display_labels=classes,
        xticks_rotation=45,
        cmap=plt.cm.Reds,
        ax=ax,
    )
    fig.tight_layout()
    disp.plot()
    fig.savefig(inference_data_path / "confusion_matrix.png")
    plt.clf()
    logging.info("Inference workflow finished successfully.")


if __name__ == "__main__":
    config = load_inference_config()
    run_inference(config)
