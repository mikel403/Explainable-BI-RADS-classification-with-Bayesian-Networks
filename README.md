# Interpretable Models for BI-RADS Malignancy Classification on Ultrasound

This repository contains the Bayesian network and Python code developed for
interpretable BI-RADS malignancy classification from breast ultrasound
descriptors.

The model estimates the BI-RADS assessment category from structured ultrasound
descriptors and provides an interpretable probabilistic representation of the
relationships between lesion characteristics and malignancy assessment.

This repository accompanies the manuscript:

> **Interpretable models for BI-RADS malignancy classification on ultrasound**  
> Mikel Carrilero-Mardones, Jorge Pérez-Martín, Ana Delgado-Laguna,
> and Francisco Javier Díez.  
> *International Journal of Intelligent Systems*. Manuscript under review.

## Bayesian network

The Bayesian network estimates the BI-RADS assessment category from the
following breast ultrasound descriptors:

- Shape
- Margin
- Orientation
- Echogenicity
- Posterior acoustic features
- Calcifications
- Special cases

The possible output categories are:

- BI-RADS 2
- BI-RADS 3
- BI-RADS 4A
- BI-RADS 4B
- BI-RADS 4C
- BI-RADS 5

The network structure is similar to a Naive Bayes classifier, with an additional
dependency from **shape to orientation**. This dependency reflects the
relationship between these two descriptors: round lesions have no orientation,
whereas most oval lesions have a parallel orientation.

The conditional probability distributions are estimated from the training data
using maximum likelihood estimation with additive smoothing:

```text
alpha = 0.5
```

A uniform prior distribution is assigned to the BI-RADS node. Therefore, the
classification is driven by the observed ultrasound descriptors rather than by
the prevalence of the BI-RADS categories in the training dataset.

## Repository structure

```text
BayesianNetwork/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── model/
│   ├── birads_bayesian_network.pgmx
│   └── birads_bayesian_network.pkl
│
└── src/
    ├── training_and_cross_validation.py
    └── inference.py
```

## OpenMarkov model

The Bayesian network is provided in OpenMarkov format:

```text
model/birads_bayesian_network.pgmx
```

The model can be opened with **OpenMarkov**, an open-source software tool for
probabilistic graphical models developed at the Research Centre for Intelligent
Decision-Support Systems (CISIAD) of the Universidad Nacional de Educación a
Distancia (UNED).

OpenMarkov allows users to graphically inspect the network structure, examine
its conditional probability distributions, introduce evidence for the BI-RADS
descriptors, and perform probabilistic inference interactively.

This is particularly useful for exploring the explainability of the model:
users can modify individual descriptors and observe how these changes affect
the probability distribution over BI-RADS categories.

**OpenMarkov website and download:**  
https://www.openmarkov.org/index.php?lang=en

Documentation and tutorials are also available from the OpenMarkov website.

## Python installation

The Python implementation was developed using **Python 3.7.13**.

The required packages are listed in `requirements.txt`:

```text
numpy==1.19.5
pandas==1.3.5
pgmpy==0.1.22
scikit-learn==1.0.2
```

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Training and cross-validation

The training and evaluation procedure used in the manuscript is implemented in:

```text
src/training_and_cross_validation.py
```

The script performs 10-fold cross-validation. In each fold, the conditional
probability distributions are estimated exclusively from the training
partition and additive smoothing with `alpha = 0.5` is applied.

The BI-RADS prior is subsequently set to a uniform distribution.

The evaluation includes:

- Accuracy
- Cohen's kappa
- Precision for each BI-RADS category
- Recall for each BI-RADS category
- F1-score for each BI-RADS category
- Confusion matrix

The script can be executed with:

```bash
python src/training_and_cross_validation.py
```

The complete dataset used in the experiments reported in the manuscript is not
distributed with this repository due to data-sharing restrictions. Therefore,
the exact experimental results cannot be reproduced without access to the
complete study dataset.

## Inference

A trained version of the Bayesian network is provided for Python inference:

```text
model/birads_bayesian_network.pkl
```

An example of inference on a new case is provided in:

```text
src/inference.py
```

The script can be executed with:

```bash
python src/inference.py
```

For example, a breast lesion can be represented using its BI-RADS descriptors:

```python
case = {
    "shape": "oval",
    "margin": "circumscribed",
    "orientation": "parallel",
    "echogenicity": "hypoechoic",
    "posterior": "no features",
    "calcifications": "no calcifications",
    "special_cases": "other",
}
```

The model returns the posterior probability associated with each BI-RADS
category and identifies the category with the highest probability.

In addition to the final prediction, the complete probability distribution can
be inspected to analyse the uncertainty associated with the assessment.

## Data availability

The complete dataset used to train and evaluate the Bayesian network cannot be
publicly distributed due to data-sharing restrictions.

However, the BI-RADS descriptions corresponding to the BCD dataset have been
made publicly available through **e-cienciaDatos**. The dataset contains
structured annotations provided by three radiologists and can be used to
explore the model with real BI-RADS descriptor configurations.

**Dataset DOI:**  
https://doi.org/10.21950/Z5QHRN

Please note that this dataset represents only part of the data used in the
study and therefore cannot be used to reproduce the complete experimental
results reported in the manuscript.

## Results

Using 10-fold cross-validation, the Bayesian network achieved:

- **Accuracy:** 0.731 ± 0.043
- **Cohen's kappa:** 0.674 ± 0.051
- **Mean F1-score:** 0.71

Most classification errors occurred between adjacent BI-RADS categories.

The Bayesian network achieved the highest accuracy and Cohen's kappa among the
seven evaluated models, although the differences between models were not
statistically significant.

Beyond predictive performance, the Bayesian network provides an interpretable
representation of the BI-RADS classification process. Its conditional
probabilities can be used to analyse the influence of individual descriptors,
explain the reasoning behind a particular classification, and perform
hypothetical or what-if reasoning by modifying the observed evidence.

## Reproducibility

This repository provides:

- The Bayesian network in OpenMarkov (`.pgmx`) format.
- The trained Bayesian network for Python inference (`.pkl`).
- The implementation of the training and 10-fold cross-validation procedure.
- The additive smoothing procedure used to estimate the conditional
  probabilities (`alpha = 0.5`).
- Python code for inference on new BI-RADS descriptor configurations.
- Access to the publicly available BI-RADS annotations from the BCD dataset.

The complete study dataset is not redistributed due to data-sharing
restrictions.

## Citation

The manuscript associated with this repository is currently under review:

> Carrilero-Mardones, M., Pérez-Martín, J., Delgado-Laguna, A., & Díez, F. J.  
> **Interpretable models for BI-RADS malignancy classification on ultrasound.**  
> *International Journal of Intelligent Systems*. Manuscript under review.

The complete bibliographic reference and DOI will be added after publication.

If you use the publicly available BI-RADS annotation dataset, please also cite
the dataset using its DOI:

https://doi.org/10.21950/Z5QHRN

## License

The source code in this repository is released under the MIT License.
See the `LICENSE` file for details.

## Disclaimer

This repository is provided for research and educational purposes only.

The Bayesian network and associated software are research prototypes and have
not been validated or approved for clinical use. They are not intended to
provide medical advice, establish a diagnosis, or replace the judgement of
qualified healthcare professionals.

The predictions generated by the model should not be used for clinical
decision-making or patient management.

The publicly available dataset referenced in this repository is distributed
separately and is subject to the terms and license specified in its original
data repository.