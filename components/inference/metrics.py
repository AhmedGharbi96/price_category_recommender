import itertools
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score

from components.inference.inferrer import Inferrer


def micro_recall_score(
    model: Inferrer,
    x: pd.DataFrame,
    y: pd.DataFrame,
    per_class_score: Optional[bool] = False,
):
    """When per_class_score is True, returns the recall score per class,
    else, performs micro-averaging.
    In micro averaging, the recall is calculated by summing up the
    true positive, false positive, and false negative counts over all
    classes and then computing the overall recall from these aggregated
    counts. Micro averaging is recommended for imbalanced class distributions,
    as it emphasizes the performance on the minority classes as much as
    on the majority classes.

    Args:
        model (Inferrer): model
        x (pd.DataFrame): feature df
        y (pd.DataFrame): ground truth
        per_class_score (Optional[bool], optional): Defaults to False.

    Returns:
        Union[float, np.ndarray[float]]: if per_class_score is true,
        returns the micro average, else returns recall per class.
    """
    average = None
    if not per_class_score:
        average = "micro"
    preds = model.predict(x)
    return recall_score(preds, y, average=average)


def soft_micro_recall_score(
    model: Inferrer,
    x: pd.DataFrame,
    y: pd.DataFrame,
    per_class_score: Optional[bool] = False,
):
    """This is a softer version of the micro_recall_score. The model
    suggests two ranges for the client, if one the ranges is correct,
    we consider the model prediction to be correct.

    Args:
        model (Inferrer): model
        x (pd.DataFrame): feature df
        y (pd.DataFrame): ground truth
        per_class_score (Optional[bool], optional): Defaults to False.

    Returns:
        Union[float, np.ndarray[float]]: if per_class_score is true,
        returns the micro average, else returns recall per class.
    """
    average = None
    if not per_class_score:
        average = "micro"
    preds = model.predict_top_2_for_metrics(x, y)
    return recall_score(preds, y, average=average)


def micro_precision_score(
    model: Inferrer,
    x: pd.DataFrame,
    y: pd.DataFrame,
    per_class_score: Optional[bool] = False,
):
    """When per_class_score is True, returns the precision score per class,
    else, performs micro-averaging.
    In micro averaging, the precision is calculated by summing up the
    true positive, false positive, and false negative counts over all
    classes and then computing the overall precision from these aggregated
    counts. Micro averaging is recommended for imbalanced class distributions,
    as it emphasizes the performance on the minority classes as much as
    on the majority classes.

    Args:
        model (Inferrer): model
        x (pd.DataFrame): feature df
        y (pd.DataFrame): ground truth
        per_class_score (Optional[bool], optional): Defaults to False.

    Returns:
        Union[float, np.ndarray[float]]: if per_class_score is true,
        returns the micro average, else returns precision per class.
    """
    average = None
    if not per_class_score:
        average = "micro"
    preds = model.predict(x)
    return precision_score(preds, y, average=average)


def soft_micro_precision_score(
    model: Inferrer,
    x: pd.DataFrame,
    y: pd.DataFrame,
    per_class_score: Optional[bool] = False,
):
    """This is a softer version of the micro_precision_score. The model
    suggests two ranges for the client, if one the ranges is correct,
    we consider the model prediction to be correct.

    Args:
        model (Inferrer): model
        x (pd.DataFrame): feature df
        y (pd.DataFrame): ground truth
        per_class_score (Optional[bool], optional): Defaults to False.

    Returns:
        Union[float, np.ndarray[float]]: if per_class_score is true,
        returns the micro average, else returns precision per class.
    """
    average = None
    if not per_class_score:
        average = "micro"
    preds = model.predict_top_2_for_metrics(x, y)
    return precision_score(preds, y, average=average)


def micro_f1_score(
    model: Inferrer,
    x: pd.DataFrame,
    y: pd.DataFrame,
    per_class_score: Optional[bool] = False,
):
    """When per_class_score is True, returns the f1 score per class,
    else, performs micro-averaging.
    In micro averaging, f1 is calculated by summing up the
    true positive, false positive, and false negative counts over all
    classes and then computing the overall f1 from these aggregated
    counts. Micro averaging is recommended for imbalanced class distributions,
    as it emphasizes the performance on the minority classes as much as
    on the majority classes.

    Args:
        model (Inferrer): model
        x (pd.DataFrame): feature df
        y (pd.DataFrame): ground truth
        per_class_score (Optional[bool], optional): Defaults to False.

    Returns:
        Union[float, np.ndarray[float]]: if per_class_score is true,
        returns the micro average, else returns f1 per class.
    """
    average = None
    if not per_class_score:
        average = "micro"
    preds = model.predict(x)
    return f1_score(preds, y, average=average)


def soft_micro_f1_score(
    model: Inferrer,
    x: pd.DataFrame,
    y: pd.DataFrame,
    per_class_score: Optional[bool] = False,
):
    """This is a softer version of the micro_f1_score. The model
    suggests two ranges for the client, if one the ranges is correct,
    we consider the model prediction to be correct.

    Args:
        model (Inferrer): model
        x (pd.DataFrame): feature df
        y (pd.DataFrame): ground truth
        per_class_score (Optional[bool], optional): Defaults to False.

    Returns:
        Union[float, np.ndarray[float]]: if per_class_score is true,
        returns the micro average, else returns f1 per class.
    """
    average = None
    if not per_class_score:
        average = "micro"
    preds = model.predict_top_2_for_metrics(x, y)
    return f1_score(preds, y, average=average)


def weighted_cohen_kappa_score(
    model: Inferrer,
    x: pd.DataFrame,
    y: pd.DataFrame,
):
    """Weighted Cohen's kappa is a measure of the agreement between
      two ordinally scaled samples. In this use case, it represents
      the agreement between the model predictions and the ground truth.

      The intuition behind using this metric is the following: the model's
      false predictions should not all be treated the same. For example,
      if the ground-truth label is 4, model_1 predicts 2 and model_2 predicts 3,
      this metric considers model_2 to be superior than model_1 because 3 is
      closer to 4 than 2.

      We assumed there is order in the price_category. The higher the category,
      the higher the range is. We concluded this after observing a high
      correlation between the price_category and the totalArea feature.
      Generally, the higher the total area of a real-estate, the higher the price.

    Args:
        model (Inferrer): model
        x (pd.DataFrame): feature df
        y (pd.DataFrame): ground truth
    """
    preds = model.predict(x)
    return cohen_kappa_score(preds, y, weights="linear")


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion Matrix", cmap=plt.cm.Reds
):
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], ".2f" if normalize else "d"),
            horizontalalignment="center",
            color="black" if cm[i, j] > sum(cm[:, j]) / 2.0 else "red",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
