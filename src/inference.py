"""
Inference with the trained Bayesian network for BI-RADS assessment.

The script loads the final Bayesian network trained with the complete
dataset and estimates the posterior probability of each BI-RADS category
from a set of breast ultrasound descriptors.
"""

from pathlib import Path
import pickle

import numpy as np


# =============================================================================
# Paths
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "model" / "birads_bayesian_network.pkl"


# =============================================================================
# BI-RADS descriptor values
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
}


# =============================================================================
# Model loading
# =============================================================================

def load_model(model_path=MODEL_PATH):
    """Load the trained Bayesian network."""

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run training_and_cross_validation.py first."
        )

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    return model


# =============================================================================
# Input validation and encoding
# =============================================================================

def validate_case(case):
    """Check that all required descriptors contain valid values."""

    expected_descriptors = set(DESCRIPTOR_VALUES.keys())
    provided_descriptors = set(case.keys())

    missing = expected_descriptors - provided_descriptors

    if missing:
        raise ValueError(
            f"Missing descriptors: {sorted(missing)}"
        )

    unexpected = provided_descriptors - expected_descriptors

    if unexpected:
        raise ValueError(
            f"Unexpected descriptors: {sorted(unexpected)}"
        )

    for descriptor, value in case.items():

        valid_values = DESCRIPTOR_VALUES[descriptor]

        if value not in valid_values:
            raise ValueError(
                f"Invalid value '{value}' for '{descriptor}'. "
                f"Valid values are: {valid_values}"
            )


def encode_case(case):
    """Convert descriptor labels into the integer states used by the model."""

    validate_case(case)

    encoded_case = {}

    for descriptor, value in case.items():

        encoded_case[descriptor] = (
            DESCRIPTOR_VALUES[descriptor].index(value)
        )

    return encoded_case


# =============================================================================
# Inference
# =============================================================================

def predict_birads(case, model):
    """
    Estimate the posterior probability of each BI-RADS category.

    Parameters
    ----------
    case : dict
        Dictionary containing the BI-RADS ultrasound descriptors.

    model
        Trained pgmpy Bayesian network.

    Returns
    -------
    dict
        Posterior probability for each BI-RADS category.
    """

    encoded_case = encode_case(case)

    joint_probabilities = []

    for birads_state in range(len(BIRADS_VALUES)):

        complete_state = dict(encoded_case)
        complete_state["BIRADS"] = birads_state

        probability = model.get_state_probability(
            complete_state
        )

        joint_probabilities.append(probability)

    joint_probabilities = np.asarray(
        joint_probabilities,
        dtype=float,
    )

    total_probability = joint_probabilities.sum()

    if total_probability == 0:
        raise ValueError(
            "The model assigned zero probability to the provided evidence."
        )

    posterior_probabilities = (
        joint_probabilities / total_probability
    )

    return {
        birads: probability
        for birads, probability
        in zip(
            BIRADS_VALUES,
            posterior_probabilities,
        )
    }


def print_prediction(probabilities):
    """Print BI-RADS probabilities and the most probable category."""

    predicted_birads = max(
        probabilities,
        key=probabilities.get,
    )

    print("\nBI-RADS probabilities")
    print("---------------------")

    for birads, probability in probabilities.items():
        print(
            f"BI-RADS {birads:>2}: "
            f"{probability:.4f} "
            f"({probability * 100:.2f}%)"
        )

    print(
        f"\nPredicted BI-RADS: {predicted_birads}"
    )


# =============================================================================
# Example
# =============================================================================

def main():

    model = load_model()

    example_case = {
        "shape": "oval",
        "margin": "circumscribed",
        "orientation": "parallel",
        "echogenicity": "hypoechoic",
        "posterior": "no features",
        "calcifications": "no calcifications",
        "special_cases": "other",
    }

    print("Example case")
    print("------------")

    for descriptor, value in example_case.items():
        print(
            f"{descriptor}: {value}"
        )

    probabilities = predict_birads(
        example_case,
        model,
    )

    print_prediction(
        probabilities
    )


if __name__ == "__main__":
    main()