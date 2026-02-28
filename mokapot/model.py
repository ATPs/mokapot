"""mokapot implements an algorithm for training machine learning models to
distinguish high-scoring target peptide-spectrum matches (PSMs) from decoy PSMs
using an iterative procedure. It is the :py:class:`Model` class that contains
this logic. A :py:class:`Model` instance can be created from any object with a
`scikit-learn estimator interface
<https://scikit-learn.org/stable/developers/develop.html>`_, allowing a wide
variety of models to be used. Once initialized, the :py:meth:`Model.fit` method
trains the underyling classifier using :doc:`a collection of PSMs <dataset>`
with this iterative approach.

Additional subclasses of the :py:class:`Model` class are available for
typical use cases. For example, use :py:class:`PercolatorModel` if you
want to emulate the behavior of Percolator.

"""

import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection._search import BaseSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from typeguard import typechecked

from mokapot import LinearPsmDataset
from mokapot.dataset.base import update_labels

LOGGER = logging.getLogger(__name__)

# Constants -------------------------------------------------------------------
PERC_GRID = {
    "class_weight": [
        {0: neg, 1: pos} for neg in (0.1, 1, 10) for pos in (0.1, 1, 10)
    ]
}

# Errors ----------------------------------------------------------------------


class BestFeatureIsBetterError(RuntimeError):
    """Raised when the best feature is better than the model."""

    pass


class ModelIterationError(RuntimeError):
    """Raised when the model does not improve after training."""

    pass


class PercolatorWeightsParseError(ValueError):
    """Raised when a file cannot be parsed as Percolator weights."""

    pass


# Classes ---------------------------------------------------------------------
@typechecked
class Model:
    """
    A machine learning model to re-score PSMs.

    Any classifier with a `scikit-learn estimator interface
    <https://scikit-learn.org/stable/developers/develop.html#estimators>`_
    can be used. This class also supports hyper parameter optimization
    using classes from the :py:mod:`sklearn.model_selection`
    module, such as the :py:class:`~sklearn.model_selection.GridSearchCV`
    and :py:class:`~sklearn.model_selection.RandomizedSearchCV` classes.

    Parameters
    ----------
    estimator : classifier object
        A classifier that is assumed to implement the scikit-learn
        estimator interface. To emulate Percolator (an SVM model) use
        :py:class:`PercolatorModel` instead.
    scaler : scaler object or "as-is", optional
        Defines how features are normalized before model fitting and
        prediction. The default, :code:`None`, subtracts the mean and scales
        to unit variance using
        :py:class:`sklearn.preprocessing.StandardScaler`.
        Other scalers should follow the `scikit-learn transformer
        interface
        <https://scikit-learn.org/stable/developers/develop.html#apis-of-scikit-learn-objects>`_
        , implementing :code:`fit_transform()` and :code:`transform()` methods.
        Alternatively, the string :code:`"as-is"` leaves the features in
        their original scale.
    train_fdr : float, optional
        The maximum false discovery rate at which to consider a target PSM as a
        positive example.
    max_iter : int, optional
        The number of iterations to perform.
    direction : str or None, optional
        The name of the feature to use as the initial direction for ranking
        PSMs. The default, :code:`None`, automatically
        selects the feature that finds the most PSMs below the
        `train_fdr`. This
        will be ignored in the case the model is already trained.
    override : bool, optional
        If the learned model performs worse than the best feature, should
        the model still be used?
    shuffle : bool, optional
        Should the order of PSMs be randomized for training? For deterministic
        algorithms, this will have no effect.
    rng : int or numpy.random.Generator, optional
        The seed or generator used for model training.

    Attributes
    ----------
    estimator : classifier object
        The classifier used to re-score PSMs.
    scaler : scaler object
        The scaler used to normalize features.
    features : list of str or None
        The name of the features used to fit the model. None if the
        model has yet to be trained.
    is_trained : bool
        Indicates if the model has been trained.
    train_fdr : float
        The maximum false discovery rate at which to consider a target PSM as a
        positive example.
    max_iter : int
        The number of iterations to perform.
    direction : str or None
        The name of the feature to use as the initial direction for ranking
        PSMs.
    override : bool
        If the learned model performs worse than the best feature, should
        the model still be used?
    shuffle : bool
        Is the order of PSMs shuffled for training?
    fold : int or None
        The CV fold on which this model was fit, if any.
    rng : numpy.random.Generator
        The random number generator.
    """

    def __init__(
        self,
        estimator,
        scaler=None,
        train_fdr=0.01,
        max_iter=10,
        direction=None,
        override=False,
        shuffle=True,
        rng=None,
    ):
        """Initialize a Model object"""
        self.estimator = clone(estimator)
        self.features = None
        self.is_trained = False
        self.feat_pass = None
        self.best_feat = None
        self.desc = None

        if scaler == "as-is":
            self.scaler = DummyScaler()
        elif scaler is None:
            self.scaler = StandardScaler()
        else:
            self.scaler = clone(scaler)

        self.train_fdr = train_fdr
        self.max_iter = max_iter
        self.direction = direction
        self.override = override
        self.shuffle = shuffle
        self.rng = rng

        # To keep track of the fold that this was trained on.
        # Needed to ensure reproducibility in brew() with
        # multiprocessing.
        self.fold = None

        # Sort out whether we need to optimize hyperparameters:
        if isinstance(self.estimator, BaseSearchCV):
            self._needs_cv = True
        else:
            self._needs_cv = False

    def __repr__(self):
        """How to print the class"""
        trained = {True: "A trained", False: "An untrained"}
        return (
            f"{trained[self.is_trained]} mokapot.model.Model object:\n"
            f"\testimator: {self.estimator}\n"
            f"\tscaler: {self.scaler}\n"
            f"\tfeatures: {self.features}"
        )

    @property
    def rng(self):
        """The random number generator for model training."""
        return self._rng

    @rng.setter
    def rng(self, rng):
        """Set the random number generator"""
        self._rng = np.random.default_rng(rng)

    @typechecked
    def save(self, out_file: Path):
        """
        Save the model to a file.

        Parameters
        ----------
        out_file : str
            The name of the file for the saved model.

        Returns
        -------
        str
            The output file name.

        Notes
        -----
        Because classes may change between mokapot and scikit-learn
        versions, a saved model may not work when either is changed
        from the version that created the model.
        """
        with open(out_file, "wb+") as out:
            pickle.dump(self, out)

        return out_file

    def decision_function(self, dataset: LinearPsmDataset):
        """
        Score a collection of PSMs

        Parameters
        ----------
        dataset : PsmDataset object
            :doc:`A collection of PSMs <dataset>` to score.

        Returns
        -------
        numpy.ndarray
            A :py:class:`numpy.ndarray` containing the score for each PSM.
        """
        # Q: we should rename this methid to "score_dataset" ...
        # ... or just remove it ... since it is redundant with `predict`
        if not self.is_trained:
            raise NotFittedError("This model is untrained. Run fit() first.")

        feat_names = dataset.features.columns.tolist()
        if set(feat_names) != set(self.features):
            raise ValueError(
                "Features of the input data do not "
                "match the features of this Model."
            )

        feat = self.scaler.transform(
            dataset.features.loc[:, self.features].values
        )

        return _get_scores(self.estimator, feat)

    def predict(self, dataset: LinearPsmDataset):
        """Alias for :py:meth:`decision_function`."""
        return self.decision_function(dataset)

    def fit(self, dataset: LinearPsmDataset):
        """
        Fit the model using the Percolator algorithm.

        The model if trained by iteratively learning to separate decoy
        PSMs from high-scoring target PSMs. By default, an initial
        direction is chosen as the feature that best separates target
        from decoy PSMs. A false discovery rate threshold is used to
        define how high a target must score to be used as a positive
        example in the next training iteration.

        Parameters
        ----------
        dataset : PsmDataset object
            :doc:`A collection of PSMs <dataset>` from which to train
            the model.

        Returns
        -------
        self
        """
        if not (dataset.targets).sum():
            raise ValueError("No target PSMs were available for training.")

        if not (~dataset.targets).sum():
            raise ValueError("No decoy PSMs were available for training.")

        if len(dataset.data) <= 200:
            LOGGER.warning(
                "Few PSMs are available for model training (%i). "
                "The learned models may be unstable.",
                len(dataset.data),
            )

        # Choose the initial direction
        (
            start_labels,
            self.feat_pass,
            self.best_feat,
            self.desc,
        ) = _get_starting_labels(dataset, self)

        # Normalize Features
        self.features = dataset.features.columns.tolist()
        norm_feat = self.scaler.fit_transform(dataset.features.values)

        # Shuffle order
        shuffled_idx = self.rng.permutation(np.arange(len(start_labels)))
        original_idx = np.argsort(shuffled_idx)
        if self.shuffle:
            norm_feat = norm_feat[shuffled_idx, :]
            start_labels = start_labels[shuffled_idx]

        # Prepare the model:
        model = _find_hyperparameters(self, norm_feat, start_labels)

        # Begin training loop
        target = start_labels
        num_passed = []
        LOGGER.info("Beginning training loop...")
        for i in range(self.max_iter):
            # Fit the model
            samples = norm_feat[target.astype(bool), :]
            iter_targ = (target[target.astype(bool)] + 1) / 2
            model.fit(samples, iter_targ)

            # Update scores
            scores = _get_scores(model, norm_feat)
            scores = scores[original_idx]

            # Update target
            target = update_labels(
                scores, dataset.target_values, eval_fdr=self.train_fdr
            )
            target = target[shuffled_idx]
            num_passed.append((target == 1).sum())

            LOGGER.debug(
                "\t- Iteration %i: %i training PSMs passed.", i, num_passed[i]
            )

            if num_passed[i] == 0:
                raise ModelIterationError(
                    "Model performs worse after training."
                )

        # If the model performs worse than what was initialized:
        best_feat_better = num_passed[-1] <= self.feat_pass
        start_better = num_passed[-1] <= (start_labels == 1).sum()

        if best_feat_better or start_better:
            if self.override:
                LOGGER.warning("Model performs worse after training.")
            else:
                raise BestFeatureIsBetterError(
                    "Model performs worse after training."
                )

        self.estimator = model
        weights = _get_weights(self.estimator, self.features)
        if weights is not None:
            LOGGER.debug("Normalized feature weights in the learned model:")
            for line in weights:
                LOGGER.debug("    %s", line)

        self.is_trained = True
        LOGGER.info("Done training.")
        return self


class PercolatorModel(Model):
    """
    A model that emulates Percolator.
    Create linear support vector machine (SVM) model that is similar
    to the one used by Percolator. This is the default model used by
    mokapot.

    Parameters
    ----------
    scaler : scaler object or "as-is", optional
        Defines how features are normalized before model fitting and
        prediction. The default, :code:`None`, subtracts the mean and scales
        to unit variance using
        :py:class:`sklearn.preprocessing.StandardScaler`.
        Other scalers should follow the `scikit-learn transformer
        interface
        <https://scikit-learn.org/stable/developers/develop.html#apis-of-scikit-learn-objects>`_
        , implementing :code:`fit_transform()` and :code:`transform()` methods.
        Alternatively, the string :code:`"as-is"` leaves the features in
        their original scale.
    train_fdr : float, optional
        The maximum false discovery rate at which to consider a target PSM as a
        positive example.
    max_iter : int, optional
        The number of iterations to perform.
    direction : str or None, optional
        The name of the feature to use as the initial direction for ranking
        PSMs. The default, :code:`None`, automatically
        selects the feature that finds the most PSMs below the
        `train_fdr`. This
        will be ignored in the case the model is already trained.
    override : bool, optional
        If the learned model performs worse than the best feature, should
        the model still be used?
    n_jobs : int, optional
        The number of jobs used to parallelize the hyperparameter grid search.
    rng : int or numpy.random.Generator, optional
        The seed or generator used for model training.

    Attributes
    ----------
    estimator : classifier object
        The classifier used to re-score PSMs.
    scaler : scaler object
        The scaler used to normalize features.
    features : list of str or None
        The name of the features used to fit the model. None if the
        model has yet to be trained.
    is_trained : bool
        Indicates if the model has been trained.
    train_fdr : float
        The maximum false discovery rate at which to consider a target PSM as a
        positive example.
    max_iter : int
        The number of iterations to perform.
    direction : str or None
        The name of the feature to use as the initial direction for ranking
        PSMs.
    override : bool
        If the learned model performs worse than the best feature, should
        the model still be used?
    n_jobs : int
        The number of jobs to use for parallizing the hyperparameter
        grid search.
    rng : numpy.random.Generator
        The random number generator.
    """

    def __init__(
        self,
        scaler=None,
        train_fdr=0.01,
        max_iter=10,
        direction=None,
        override=False,
        n_jobs=1,
        rng=None,
    ):
        """Initialize a PercolatorModel"""
        self.n_jobs = n_jobs
        rng = np.random.default_rng(rng)
        svm_model = LinearSVC(dual=False, random_state=7)
        estimator = GridSearchCV(
            svm_model,
            param_grid=PERC_GRID,
            refit=False,
            cv=KFold(3, shuffle=True, random_state=rng.integers(1, 1e6)),
            n_jobs=n_jobs,
        )

        super().__init__(
            estimator=estimator,
            scaler=scaler,
            train_fdr=train_fdr,
            max_iter=max_iter,
            direction=direction,
            override=override,
            rng=rng,
        )


class DummyScaler:
    """
    Implements the interface of scikit-learn scalers, but does
    nothing to the data. This simplifies the training code.

    :meta private:
    """

    def fit(self, x):
        pass

    def fit_transform(self, x):
        return x

    def transform(self, x):
        return x


# Functions -------------------------------------------------------------------
@typechecked
def save_model(model, out_file: Path):
    """
    Save a :py:class:`mokapot.model.Model` object to a file.

    Parameters
    ----------
    out_file : str
        The name of the file for the saved model.

    Returns
    -------
    str
        The output file name.

    Notes
    -----
    Because classes may change between mokapot and scikit-learn versions,
    a saved model may not work when either is changed from the version
    that created the model.
    """
    return model.save(out_file)


@typechecked
def save_percolator_models(models: list[Model], out_file: Path):
    """
    Save trained models as a single Percolator-style weights file.

    Parameters
    ----------
    models : list[mokapot.model.Model]
        Trained models, one for each cross-validation fold.
    out_file : str
        Output path for the Percolator-style weights file.

    Returns
    -------
    str
        The output file name.
    """
    if not models:
        raise ValueError("At least one model is required to save weights.")

    feature_names = None
    with open(out_file, "w", encoding="utf-8") as stream:
        stream.write(
            "# This file contains the weights from each cross validation "
            "bin from percolator training\n"
        )
        stream.write(
            "# First line is the feature names, followed by normalized "
            "weights, and the raw weights of bin 1\n"
        )
        stream.write("# This is repeated for the other bins\n")

        for model in models:
            features, norm_row, raw_row = _extract_weight_rows(model)
            if feature_names is None:
                feature_names = features
            elif features != feature_names:
                raise ValueError(
                    "All models must use the same features to be saved as "
                    "a single Percolator weights file."
                )

            stream.write("\t".join([*features, "m0"]) + "\n")
            stream.write(_format_weight_row(norm_row) + "\n")
            stream.write(_format_weight_row(raw_row) + "\n")

    return out_file


@typechecked
def load_models(model_file: Path):
    """
    Load one or more models for mokapot.

    Parameters
    ----------
    model_file : str
        File containing either pickled mokapot models or Percolator weights.

    Returns
    -------
    list[mokapot.model.Model]
        One or more loaded :py:class:`mokapot.model.Model` objects.

    Warnings
    --------
    Unpickling data in Python is unsafe. Make sure that the model is from
    a source that you trust.
    """
    try:
        models = _load_percolator_models(model_file)
        LOGGER.info("Loading %i Percolator model(s).", len(models))
        return models
    except (PercolatorWeightsParseError, UnicodeDecodeError):
        LOGGER.info("Loading mokapot model.")
        with open(model_file, "rb") as mod_in:
            model = pickle.load(mod_in)
        return [model]


@typechecked
def load_model(model_file: Path):
    """
    Load a saved model for mokapot.

    The saved model can either be a saved :py:class:`~mokapot.model.Model`
    object or the output model weights from Percolator. In Percolator,
    these can be obtained using the :code:`--weights` argument.

    Parameters
    ----------
    model_file : str
        The name of file from which to load the model.

    Returns
    -------
    mokapot.model.Model
        The loaded :py:class:`mokapot.model.Model` object.

    Warnings
    --------
    Unpickling data in Python is unsafe. Make sure that the model is from
    a source that you trust.
    """
    models = load_models(model_file)
    if len(models) > 1:
        LOGGER.info(
            "Loaded %i models from '%s'; returning the first model.",
            len(models),
            model_file,
        )
    return models[0]


# Private Functions -----------------------------------------------------------
def _extract_weight_rows(
    model: Model,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Extract normalized and raw weights for Percolator-style output."""
    if not model.is_trained:
        raise ValueError("Can only save trained models.")

    if model.features is None:
        raise ValueError("Model has no feature names.")

    features = list(model.features)
    try:
        weights = np.asarray(model.estimator.coef_, dtype=float).reshape(-1)
        intercept = float(np.asarray(model.estimator.intercept_)[0])
    except (AttributeError, IndexError, ValueError) as exc:
        raise ValueError(
            "Model does not expose linear coefficients/intercept."
        ) from exc

    if len(weights) != len(features):
        raise ValueError(
            "The number of model coefficients does not match the number of "
            "features."
        )

    norm_row = np.concatenate([weights, np.array([intercept])])
    raw_weights, raw_intercept = _unnormalize_weights(weights, intercept, model)
    raw_row = np.concatenate([raw_weights, np.array([raw_intercept])])
    return features, norm_row, raw_row


def _unnormalize_weights(
    weights: np.ndarray, intercept: float, model: Model
) -> tuple[np.ndarray, float]:
    """Unnormalize coefficients based on the fitted model scaler."""
    scaler = model.scaler
    if isinstance(scaler, DummyScaler):
        return weights.copy(), intercept

    if isinstance(scaler, StandardScaler):
        try:
            mean = np.asarray(scaler.mean_, dtype=float)
            scale = np.asarray(scaler.scale_, dtype=float)
        except AttributeError as exc:
            raise ValueError(
                "The model scaler is missing mean_/scale_ attributes."
            ) from exc

        if len(mean) != len(weights) or len(scale) != len(weights):
            raise ValueError(
                "Scaler mean_/scale_ length does not match model features."
            )

        safe_scale = np.where(scale == 0, 1.0, scale)
        raw_weights = weights / safe_scale
        raw_intercept = intercept - np.sum(weights * mean / safe_scale)
        return raw_weights, float(raw_intercept)

    LOGGER.warning(
        "Scaler type '%s' is not supported for unnormalizing model weights. "
        "Using normalized values as raw values.",
        type(scaler).__name__,
    )
    return weights.copy(), intercept


def _format_weight_row(weights: np.ndarray) -> str:
    """Format one line of model weights for Percolator output."""
    return "\t".join([f"{weight:.4f}" for weight in weights])


def _load_percolator_models(model_file: Path) -> list[Model]:
    """Load one or more models from a Percolator-style weights file."""
    blocks = _parse_percolator_weight_blocks(model_file)
    models = []
    for fold, (features, _, raw_weights) in enumerate(blocks):
        model = Model(estimator=LinearSVC(), scaler="as-is")
        model.estimator.coef_ = raw_weights[:-1][np.newaxis, :]
        model.estimator.intercept_ = np.array([raw_weights[-1]])
        model.features = features
        model.is_trained = True
        model.fold = fold
        model.override = True
        model.feat_pass = 0
        model.desc = True
        models.append(model)
    return models


def _parse_percolator_weight_blocks(
    model_file: Path,
) -> list[tuple[list[str], np.ndarray, np.ndarray]]:
    """
    Parse a Percolator-style model weights file.

    Returns blocks of (feature_names, normalized_weights, raw_weights).
    """
    parsed_lines = []
    with open(model_file, "r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parsed_lines.append((line_no, stripped.split("\t")))

    if not parsed_lines:
        raise PercolatorWeightsParseError(
            f"No Percolator weight data found in '{model_file}'."
        )

    blocks = []
    idx = 0
    while idx < len(parsed_lines):
        line_no, header = parsed_lines[idx]
        if not _is_weights_header(header):
            raise PercolatorWeightsParseError(
                f"Expected a Percolator header ending with 'm0' at line "
                f"{line_no} in '{model_file}'."
            )
        features = header[:-1]
        if not features:
            raise PercolatorWeightsParseError(
                f"No features found in header at line {line_no} in "
                f"'{model_file}'."
            )

        idx += 1
        if idx >= len(parsed_lines):
            raise PercolatorWeightsParseError(
                f"Missing normalized weights after header at line {line_no} "
                f"in '{model_file}'."
            )

        norm_line_no, norm_tokens = parsed_lines[idx]
        if _is_weights_header(norm_tokens):
            raise PercolatorWeightsParseError(
                f"Missing normalized weights at line {norm_line_no} in "
                f"'{model_file}'."
            )
        norm_weights = _parse_weights_row(
            tokens=norm_tokens,
            expected_len=len(header),
            line_no=norm_line_no,
            model_file=model_file,
        )
        idx += 1

        raw_weights = norm_weights.copy()
        if idx < len(parsed_lines):
            raw_line_no, raw_tokens = parsed_lines[idx]
            if not _is_weights_header(raw_tokens):
                raw_weights = _parse_weights_row(
                    tokens=raw_tokens,
                    expected_len=len(header),
                    line_no=raw_line_no,
                    model_file=model_file,
                )
                idx += 1

        blocks.append((features, norm_weights, raw_weights))

    return blocks


def _is_weights_header(tokens: list[str]) -> bool:
    """Check whether a tokenized line looks like a Percolator header."""
    return bool(tokens) and tokens[-1] == "m0"


def _parse_weights_row(
    tokens: list[str], expected_len: int, line_no: int, model_file: Path
) -> np.ndarray:
    """Parse one numeric row in a Percolator weights file."""
    if len(tokens) != expected_len:
        raise PercolatorWeightsParseError(
            f"Expected {expected_len} columns at line {line_no} in "
            f"'{model_file}', got {len(tokens)}."
        )
    try:
        return np.asarray([float(token) for token in tokens], dtype=float)
    except ValueError as exc:
        raise PercolatorWeightsParseError(
            f"Non-numeric value found at line {line_no} in '{model_file}'."
        ) from exc


def _get_starting_labels(dataset: LinearPsmDataset, model):
    """
    Get labels using the initial direction.

    Parameters
    ----------
    dataset : a collection of PSMs
        The PsmDataset object
    model : mokapot.Model
        A model object (this is likely `self`)

    Returns
    -------
    start_labels : np.array
        The starting labels for model training.
    feat_pass : int
        The number of passing PSMs with the best feature.
    """

    # Note: This function does sooo much more than getting the starting
    # labels, we should at least rename it to something more descriptive.
    # JSPP 2024-12-14
    LOGGER.debug("Finding initial direction...")
    if model.direction is None and not model.is_trained:
        feat_res = dataset.find_best_feature(model.train_fdr)
        best_feat, feat_pass, start_labels, desc = (
            feat_res.feature.name,
            feat_res.feature.positives,
            feat_res.new_labels,
            feat_res.feature.descending,
        )
        LOGGER.info(
            "\t- Selected feature '%s' with %i PSMs at q<=%g.",
            best_feat,
            feat_pass,
            model.train_fdr,
        )

    elif model.is_trained:
        try:
            scores = model.estimator.decision_function(dataset.features.values)
        except AttributeError:
            scores = model.estimator.predict_proba(dataset.features).flatten()

        start_labels = dataset._update_labels(scores, eval_fdr=model.train_fdr)
        feat_pass = (start_labels == 1).sum()
        best_feat = model.best_feat
        desc = model.desc
        LOGGER.info(
            "\t- The pretrained model found %i PSMs at q<=%g.",
            feat_pass,
            model.train_fdr,
        )

    else:
        feat = dataset.features[model.direction].values
        desc_labels = update_labels(
            feat,
            dataset.target_values,
            model.train_fdr,
            desc=True,
        )
        asc_labels = update_labels(
            feat,
            dataset.target_values,
            model.train_fdr,
            desc=False,
        )
        best_feat = feat

        desc_pass = (desc_labels == 1).sum()
        asc_pass = (asc_labels == 1).sum()
        if desc_pass >= asc_pass:
            start_labels = desc_labels
            feat_pass = desc_pass
            desc = True
        else:
            start_labels = asc_labels
            feat_pass = asc_pass
            desc = False

        LOGGER.info(
            "  - Selected feature %s with %i PSMs at q<=%g.",
            model.direction,
            (start_labels == 1).sum(),
            model.train_fdr,
        )

    if not (start_labels == 1).sum():
        raise RuntimeError(
            f"No PSMs accepted at train_fdr={model.train_fdr}. "
            "Consider changing it to a higher value."
        )

    return start_labels, feat_pass, best_feat, desc


def _find_hyperparameters(model, features, labels):
    """
    Find the hyperparameters for the model.

    Parameters
    ----------
    model : a mokapot.Model
        The model to fit.
    features : array-like
        The features to fit the model with.
    labels : array-like
        The labels for each PSM (1, 0, or -1).

    Returns
    -------
    An estimator.
    """
    if model._needs_cv:
        LOGGER.debug("Selecting hyperparameters...")
        cv_samples = features[labels.astype(bool), :]
        cv_targ = (labels[labels.astype(bool)] + 1) / 2

        # Fit the model
        model.estimator.fit(cv_samples, cv_targ)

        # Extract the best params.
        best_params = model.estimator.best_params_
        new_est = model.estimator.estimator
        new_est.set_params(**best_params)
        model._needs_cv = False
        for param, value in best_params.items():
            LOGGER.debug("\t- %s = %s", param, value)
    else:
        new_est = model.estimator

    return new_est


def _get_weights(model, features) -> list[str] | None:
    """
    If the model is a linear model, parse the weights to a list of strings.

    Parameters
    ----------
    model : estimator
        An sklearn linear_model object
    features : list of str
        The feature names, in order.

    Returns
    -------
    list of str
        The weights associated with each feature.
    """
    try:
        weights = model.coef_
        intercept = model.intercept_
        assert weights.shape[0] == 1
        assert weights.shape[1] == len(features)
        assert len(intercept) == 1
        weights = list(weights.flatten())
    except (AttributeError, AssertionError):
        LOGGER.debug("No coefficients in the current model.")
        return None

    col_width = max([len(f) for f in features]) + 2
    txt_out = ["Feature" + " " * (col_width - 7) + "Weight"]
    for weight, feature in zip(weights, features):
        space = " " * (col_width - len(feature))
        txt_out.append(feature + space + str(weight))

    txt_out.append("intercept" + " " * (col_width - 9) + str(intercept[0]))
    return txt_out


def _get_scores(model, feat):
    """Get the scores from a model

    We want to use the `decision_function` method if it is available,
    but fall back to the `predict_proba` method if it isn't. In sklearn,
    `predict_proba` for a binary classifier returns a two-column numpy array,
    where the second column is the probability we want. However,
    skorch (and other tools) sometime do this differently, returning only
    a single column. This function makes it so mokapot can work with either.

    Parameters
    ----------
    model : an estimator object
        The model to score the PSMs.
    feat : np.ndarray
        The normalized features

    Returns
    -------
    np.ndarray
        A :py:class:`numpy.ndarray` containing the score for each PSM in feat.
    """
    try:
        return model.decision_function(feat)
    except AttributeError:
        scores = model.predict_proba(feat).squeeze()
        if len(scores.shape) == 2:
            return model.predict_proba(feat)[:, 1]
        elif len(scores.shape) == 1:
            return scores
        else:
            raise RuntimeError("'predict_proba' returned too many dimensions.")
