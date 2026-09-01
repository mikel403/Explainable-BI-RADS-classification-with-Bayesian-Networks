"""
Training and cross-validation of the Bayesian network for BI-RADS
assessment from breast ultrasound descriptors.

The network models the probabilistic relationships between BI-RADS
assessment categories and ultrasound descriptors, including an additional
dependency between lesion shape and orientation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pickle

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


# =============================================================================
# Configuration
# =============================================================================

RANDOM_SEED = 1
N_FOLDS = 10
ALPHA = 0.5

np.random.seed(RANDOM_SEED)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
MODEL_DIR = ROOT_DIR / "model"

DATA_PATH = DATA_DIR / "BI-RADS_data.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# BI-RADS categories and descriptor values
# =============================================================================

SHAPE_VALUES = sorted([
    "irregular",
    "oval",
    "round",
])

MARGIN_VALUES = sorted([
    "angulated",
    "circumscribed",
    "spiculated",
    "indistinct",
    "microlobulated",
])

ORIENTATION_VALUES = sorted([
    "not parallel",
    "parallel",
    "no orientation",
])

ECHOGENICITY_VALUES = sorted([
    "anechoic",
    "heterogeneous",
    "hypoechoic",
    "isoechoic",
    "hyperechoic",
])

POSTERIOR_VALUES = sorted([
    "enhancement",
    "no features",
    "shadowing",
    "combined pattern",
])

CALCIFICATION_VALUES = sorted([
    "no calcifications",
    "calcifications",
    "no determined",
])

SPECIAL_CASE_VALUES = sorted([
    "complicated cyst",
    "simple cyst",
    "clustered microcysts",
    "other",
])

BIRADS_VALUES = [
    "2",
    "3",
    "4A",
    "4B",
    "4C",
    "5",
]


DESCRIPTOR_VALUES = {
    "shape": SHAPE_VALUES,
    "margin": MARGIN_VALUES,
    "orientation": ORIENTATION_VALUES,
    "echogenicity": ECHOGENICITY_VALUES,
    "posterior": POSTERIOR_VALUES,
    "calcifications": CALCIFICATION_VALUES,
    "special_cases": SPECIAL_CASE_VALUES,
    "BIRADS": BIRADS_VALUES,
}


# =============================================================================
# Bayesian network structure
# =============================================================================

NETWORK_EDGES = [
    ("BIRADS", "shape"),
    ("BIRADS", "margin"),
    ("BIRADS", "orientation"),
    ("BIRADS", "echogenicity"),
    ("BIRADS", "posterior"),
    ("BIRADS", "calcifications"),
    ("BIRADS", "special_cases"),
    ("shape", "orientation"),
]


def create_bayesian_network():
    """Create the Bayesian network structure used in the study."""
    return BayesianNetwork(NETWORK_EDGES)

def save_model(model, model_path):
    """Save the trained Bayesian network for later inference."""

    with open(model_path, "wb") as file:
        pickle.dump(model, file)

# =============================================================================
# Data loading and preprocessing
# =============================================================================

def load_and_preprocess_data(data_path):
    """Load the processed BI-RADS annotations and encode categorical values."""

    data = pd.read_csv(
        data_path,
        header=0,
        index_col=0,
    )

    data.columns = [
        "shape",
        "margin",
        "orientation",
        "echogenicity",
        "posterior",
        "calcifications",
        "special_cases",
        "BIRADS",
    ]

    category_to_index = {
        variable: {
            value: index
            for index, value in enumerate(values)
        }
        for variable, values in DESCRIPTOR_VALUES.items()
    }

    for column in data.columns:
        data[column] = data[column].replace(
            category_to_index[column]
        )

    data = data.astype(int)

    return data


# =============================================================================
# Prediction
# =============================================================================

def predict_probabilities(case, model):
    """
    Compute the joint probability of the observed descriptors for each
    possible BI-RADS category.
    """

    evidence = dict(case)
    probabilities = []

    for birads_state in range(len(BIRADS_VALUES)):
        evidence["BIRADS"] = birads_state
        probability = model.get_state_probability(evidence)
        probabilities.append(probability)

    return np.asarray(probabilities)


def predict_cases(cases, model):
    """Predict the most probable BI-RADS category for each case."""

    predictions = []

    for _, case in cases.iterrows():
        probabilities = predict_probabilities(case, model)
        predictions.append(np.argmax(probabilities))

    return np.asarray(predictions)


# =============================================================================
# Additive smoothing
# =============================================================================

def apply_additive_smoothing(model, train_data, alpha):
    """
    Apply additive smoothing to the conditional probability distributions.

    BI-RADS is assigned a uniform prior. Most descriptors are conditioned
    only on BI-RADS. Orientation is conditioned on both BI-RADS and shape.
    """

    smoothed_model = create_bayesian_network()

    n_birads = len(BIRADS_VALUES)

    for cpd in model.get_cpds():

        variable = cpd.variable
        n_states = len(DESCRIPTOR_VALUES[variable])

        # ---------------------------------------------------------------------
        # Uniform BI-RADS prior
        # ---------------------------------------------------------------------

        if variable == "BIRADS":

            values = np.full(
                (n_birads, 1),
                1.0 / n_birads,
            )

            new_cpd = TabularCPD(
                variable="BIRADS",
                variable_card=n_birads,
                values=values,
            )

        # ---------------------------------------------------------------------
        # P(orientation | BI-RADS, shape)
        # ---------------------------------------------------------------------

        elif variable == "orientation":

            n_orientations = len(ORIENTATION_VALUES)
            n_shapes = len(SHAPE_VALUES)

            values = np.zeros(
                (n_orientations, n_birads, n_shapes),
                dtype=float,
            )

            for birads_state in range(n_birads):

                for shape_state in range(n_shapes):

                    parent_mask = (
                        (train_data["BIRADS"] == birads_state)
                        & (train_data["shape"] == shape_state)
                    )

                    parent_count = parent_mask.sum()

                    for orientation_state in range(n_orientations):

                        joint_count = (
                            parent_mask
                            & (
                                train_data["orientation"]
                                == orientation_state
                            )
                        ).sum()

                        values[
                            orientation_state,
                            birads_state,
                            shape_state,
                        ] = (
                            joint_count + alpha
                        ) / (
                            parent_count
                            + alpha * n_orientations
                        )

            values = values.reshape(n_orientations, -1)

            new_cpd = TabularCPD(
                variable="orientation",
                variable_card=n_orientations,
                values=values,
                evidence=["BIRADS", "shape"],
                evidence_card=[n_birads, n_shapes],
            )

        # ---------------------------------------------------------------------
        # P(descriptor | BI-RADS)
        # ---------------------------------------------------------------------

        else:

            values = np.zeros(
                (n_states, n_birads),
                dtype=float,
            )

            for birads_state in range(n_birads):

                birads_mask = (
                    train_data["BIRADS"] == birads_state
                )

                parent_count = birads_mask.sum()

                for descriptor_state in range(n_states):

                    joint_count = (
                        birads_mask
                        & (
                            train_data[variable]
                            == descriptor_state
                        )
                    ).sum()

                    values[
                        descriptor_state,
                        birads_state,
                    ] = (
                        joint_count + alpha
                    ) / (
                        parent_count
                        + alpha * n_states
                    )

            new_cpd = TabularCPD(
                variable=variable,
                variable_card=n_states,
                values=values,
                evidence=["BIRADS"],
                evidence_card=[n_birads],
            )

        smoothed_model.add_cpds(new_cpd)

    return smoothed_model


# =============================================================================
# Cross-validation
# =============================================================================

def get_fold(data, fold_index, n_folds=N_FOLDS):
    """
    Split the already shuffled dataset into training and test data for
    one cross-validation fold.
    """

    fold_size = int(np.ceil(len(data) / n_folds))

    start = fold_index * fold_size
    end = min(len(data), (fold_index + 1) * fold_size)

    test_data = data.iloc[start:end]
    train_data = data.drop(test_data.index)

    return train_data, test_data


def cross_validate(data, alpha=ALPHA, n_folds=N_FOLDS):
    """Perform cross-validation and return predictions and performance."""

    all_predictions = []
    all_targets = []
    all_indices = []

    fold_accuracies = []
    fold_kappas = []

    for fold_index in range(n_folds):

        print(f"Fold {fold_index + 1}/{n_folds}")

        train_data, test_data = get_fold(
            data,
            fold_index,
            n_folds,
        )

        test_features = test_data.drop(
            columns="BIRADS"
        )

        test_targets = test_data[
            "BIRADS"
        ].to_numpy()

        model = create_bayesian_network()

        model.fit(
            train_data,
            n_jobs=1,
        )

        model = apply_additive_smoothing(
            model,
            train_data,
            alpha,
        )

        predictions = predict_cases(
            test_features,
            model,
        )

        fold_accuracies.append(
            accuracy_score(
                test_targets,
                predictions,
            )
        )

        fold_kappas.append(
            cohen_kappa_score(
                test_targets,
                predictions,
            )
        )

        all_predictions.append(predictions)
        all_targets.append(test_targets)
        all_indices.extend(test_data.index)

    all_predictions = np.concatenate(
        all_predictions
    )

    all_targets = np.concatenate(
        all_targets
    )

    return {
        "predictions": all_predictions,
        "targets": all_targets,
        "indices": np.asarray(all_indices),
        "accuracy": accuracy_score(
            all_targets,
            all_predictions,
        ),
        "accuracy_std": np.std(
            fold_accuracies
        ),
        "kappa": cohen_kappa_score(
            all_targets,
            all_predictions,
        ),
        "kappa_std": np.std(
            fold_kappas
        ),
    }

def train_final_model(data, alpha=ALPHA):
    """
    Train the final Bayesian network using the complete dataset.
    """

    model = create_bayesian_network()

    model.fit(
        data,
        n_jobs=1,
    )

    model = apply_additive_smoothing(
        model,
        data,
        alpha,
    )

    return model
# =============================================================================
# Evaluation
# =============================================================================

def calculate_classification_metrics(targets, predictions):
    """Calculate precision, recall and F1-score for each BI-RADS category."""

    labels = np.arange(len(BIRADS_VALUES))

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            targets,
            predictions,
            labels=labels,
            zero_division=0,
        )
    )

    metrics = pd.DataFrame({
        "BI-RADS": BIRADS_VALUES,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Support": support,
    })

    mean_row = pd.DataFrame({
        "BI-RADS": ["Mean"],
        "Precision": [precision.mean()],
        "Recall": [recall.mean()],
        "F1": [f1.mean()],
        "Support": [support.sum()],
    })

    return pd.concat(
        [metrics, mean_row],
        ignore_index=True,
    )


def calculate_confusion_matrix(targets, predictions):
    """Calculate the BI-RADS confusion matrix."""

    matrix = confusion_matrix(
        targets,
        predictions,
        labels=np.arange(len(BIRADS_VALUES)),
    )

    return pd.DataFrame(
        matrix,
        index=BIRADS_VALUES,
        columns=BIRADS_VALUES,
    )


def create_predictions_dataframe(indices, predictions):
    """Convert numerical BI-RADS predictions back to their labels."""

    labels = [
        BIRADS_VALUES[int(prediction)]
        for prediction in predictions
    ]

    return pd.DataFrame(
        {"BI-RADS": labels},
        index=indices,
    )


# =============================================================================
# Main
# =============================================================================

def main():

    data = load_and_preprocess_data(
        DATA_PATH
    )

    # -------------------------------------------------------------------------
    # Cross-validation
    # -------------------------------------------------------------------------

    results = cross_validate(
        data,
        alpha=ALPHA,
        n_folds=N_FOLDS,
    )

    metrics = calculate_classification_metrics(
        results["targets"],
        results["predictions"],
    )

    confusion = calculate_confusion_matrix(
        results["targets"],
        results["predictions"],
    )

    predictions = create_predictions_dataframe(
        results["indices"],
        results["predictions"],
    )

    # -------------------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------------------

    print("\nCross-validation results")
    print("------------------------")

    print(
        f"Accuracy: {results['accuracy']:.4f} "
        f"(SD: {results['accuracy_std']:.4f})"
    )

    print(
        f"Cohen's kappa: {results['kappa']:.4f} "
        f"(SD: {results['kappa_std']:.4f})"
    )

    print("\nPer-class performance")
    print("---------------------")
    print(metrics.to_string(index=False))

    print("\nConfusion matrix")
    print("----------------")
    print(confusion)

    # -------------------------------------------------------------------------
    # Save cross-validation results
    # -------------------------------------------------------------------------

    metrics.to_csv(
        RESULTS_DIR / "bayesian_network_metrics.csv",
        index=False,
    )

    confusion.to_csv(
        RESULTS_DIR / "bayesian_network_confusion_matrix.csv",
    )

    predictions.to_csv(
        RESULTS_DIR / "bayesian_network_predictions.csv",
    )
    # -------------------------------------------------------------------------
    # Train final model using the complete dataset
    # -------------------------------------------------------------------------

    final_model = train_final_model(
        data,
        alpha=ALPHA,
    )

    model_path = (
        MODEL_DIR / "birads_bayesian_network.pkl"
    )

    save_model(
        final_model,
        model_path,
    )

    print(
        f"\nFinal model saved to:\n{model_path}"
    )


if __name__ == "__main__":
    main()