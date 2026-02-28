"""Test that models work as expected"""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC

import mokapot
from mokapot import LinearPsmDataset, utils

from ..helpers.random_df import _psm_df_rand


@pytest.fixture
def psms_dataset() -> LinearPsmDataset:
    """A small LinearPsmDataset."""
    data = _psm_df_rand(
        ntargets=500,
        ndecoys=500,
        share_ids=True,
        pct_real=0.5,
        # This means there are 4 features, for decoys
        # they are all normall distributed on with mean
        # of 0 and std of 1.
        # For targets they are distributed with mean
        # of 0.5 (x3) and 2.0 (x1), and std of 1.
        score_diffs=[0.5, 0.5, 0.5, 2.0],
    )

    psms = LinearPsmDataset(
        psms=data.df,
        target_column="target",
        spectrum_columns=data.columns.spectrum_columns,
        peptide_column="peptide",
        feature_columns=data.score_cols,
        copy_data=True,
    )
    return psms


def test_model_init():
    """Test that a model initializes correctly"""
    model = mokapot.Model(
        LogisticRegression(),
        scaler=MinMaxScaler(),
        train_fdr=0.05,
        max_iter=1,
        direction="score",
        override=True,
        shuffle=False,
    )

    assert isinstance(model.estimator, LogisticRegression)
    assert isinstance(model.scaler, MinMaxScaler)
    assert model.train_fdr == 0.05
    assert model.max_iter == 1
    assert model.direction == "score"
    assert model.override
    assert not model.shuffle
    assert not model.is_trained

    model = mokapot.Model(LogisticRegression(), scaler="as-is")
    assert isinstance(model.scaler, mokapot.model.DummyScaler)

    model = mokapot.Model(LogisticRegression())
    assert isinstance(model.scaler, StandardScaler)


def test_perc_init():
    """Test the initialization of a PercolatorModel"""
    model = mokapot.PercolatorModel(
        scaler="as-is",
        train_fdr=0.05,
        max_iter=1,
        direction="score",
        override=True,
    )
    assert isinstance(model.estimator, GridSearchCV)
    assert isinstance(model.estimator.estimator, LinearSVC)
    assert isinstance(model.scaler, mokapot.model.DummyScaler)
    assert model.train_fdr == 0.05
    assert model.max_iter == 1
    assert model.direction == "score"
    assert model.override


def test_model_fit(psms_dataset):
    """Test that model fitting works"""
    model = mokapot.Model(LogisticRegression(), train_fdr=0.05, max_iter=1)
    model.fit(psms_dataset)

    assert model.is_trained

    model = mokapot.Model(LogisticRegressionCV(), train_fdr=0.05, max_iter=1)
    model.fit(psms_dataset)

    assert isinstance(model.estimator, LogisticRegression)
    assert model.is_trained

    psms_dataset._data[psms_dataset.target_column] = False
    with pytest.raises(ValueError):
        model.fit(psms_dataset)  # no targets

    psms_dataset._data[psms_dataset.target_column] = True
    with pytest.raises(ValueError):
        model.fit(psms_dataset)  # no decoys


def test_model_fit_large_subset(psms_dataset):
    model = mokapot.Model(
        LogisticRegression(),
        train_fdr=0.05,
        max_iter=1,
    )
    model.fit(psms_dataset)

    assert model.is_trained


def test_model_predict(psms_dataset):
    """Test predictions"""
    from mokapot.column_defs import ColumnGroups

    model = mokapot.Model(LogisticRegression(), train_fdr=0.05, max_iter=1)

    try:
        model.predict(psms_dataset)
    except NotFittedError:
        pass

    # The normal case
    model.fit(psms_dataset)
    scores = model.predict(psms_dataset)
    assert len(scores) == len(psms_dataset)

    # The case where a model is trained on a dataset with different features:
    # Note: 'blah' is not a feature column originally
    psms_dataset._data["blah"] = np.random.randn(len(psms_dataset))
    cgs = psms_dataset.column_groups

    new_cgs = ColumnGroups(
        columns=utils.tuplize(psms_dataset._data.columns),
        target_column=cgs.target_column,
        peptide_column=cgs.peptide_column,
        spectrum_columns=cgs.spectrum_columns,
        feature_columns=cgs.feature_columns + ("blah",),
        extra_confidence_level_columns=cgs.extra_confidence_level_columns,
        optional_columns=cgs.optional_columns,
    )
    psms_dataset._column_groups = new_cgs

    with pytest.raises(ValueError):
        model.predict(psms_dataset)


def test_model_persistance(tmp_path):
    """test that we can save and load a model"""
    model_file = tmp_path / "model.pkl"

    model = mokapot.Model(LogisticRegression(), train_fdr=0.05, max_iter=1)
    mokapot.save_model(model, model_file)
    loaded = mokapot.load_model(model_file)

    assert isinstance(loaded, mokapot.Model)


def test_dummy_scaler():
    """Test the DummyScaler class"""
    data = np.random.default_rng(42).normal(0, 1, (20, 10))
    scaler = mokapot.model.DummyScaler()
    assert (data == scaler.fit_transform(data)).all()
    assert (data == scaler.transform(data)).all()


def test_load_models_percolator_weights(tmp_path):
    """Test parsing Percolator-style weights files."""
    model_file = tmp_path / "weights.txt"
    model_file.write_text(
        "# This file contains Percolator weights\n"
        "# Header + normalized + raw for each fold\n"
        "# Repeated per fold\n"
        "feat_a\tfeat_b\tm0\n"
        "0.1000\t0.2000\t0.3000\n"
        "1.1000\t1.2000\t1.3000\n"
        "feat_a\tfeat_b\tm0\n"
        "0.4000\t0.5000\t0.6000\n"
        "1.4000\t1.5000\t1.6000\n"
    )

    loaded = mokapot.load_models(model_file)
    assert len(loaded) == 2
    assert isinstance(loaded[0].scaler, mokapot.model.DummyScaler)
    assert loaded[0].features == ["feat_a", "feat_b"]
    np.testing.assert_allclose(loaded[0].estimator.coef_, [[1.1, 1.2]])
    np.testing.assert_allclose(loaded[0].estimator.intercept_, [1.3])
    np.testing.assert_allclose(loaded[1].estimator.coef_, [[1.4, 1.5]])
    np.testing.assert_allclose(loaded[1].estimator.intercept_, [1.6])


def test_save_percolator_models_roundtrip(tmp_path):
    """Test writing Percolator-style weights and loading them back."""
    model_a = mokapot.Model(LinearSVC(), scaler="as-is")
    model_a.features = ["feat_a", "feat_b"]
    model_a.estimator.coef_ = np.array([[2.0, 4.0]])
    model_a.estimator.intercept_ = np.array([5.0])
    model_a.is_trained = True

    model_b = mokapot.Model(LinearSVC())
    model_b.features = ["feat_a", "feat_b"]
    model_b.estimator.coef_ = np.array([[2.0, 4.0]])
    model_b.estimator.intercept_ = np.array([5.0])
    model_b.scaler.mean_ = np.array([1.0, 2.0])
    model_b.scaler.scale_ = np.array([2.0, 4.0])
    model_b.is_trained = True

    model_file = tmp_path / "weights.txt"
    mokapot.save_percolator_models([model_a, model_b], model_file)

    lines = model_file.read_text().splitlines()
    assert lines[0].startswith("# This file contains the weights")
    assert lines[1].startswith("# First line is the feature names")
    assert lines[2].startswith("# This is repeated")
    assert lines[3] == "feat_a\tfeat_b\tm0"
    assert lines[6] == "feat_a\tfeat_b\tm0"
    assert len(lines) == 9

    loaded = mokapot.load_models(model_file)
    assert len(loaded) == 2

    # Raw weights for the first model are unchanged ("as-is" scaler).
    np.testing.assert_allclose(loaded[0].estimator.coef_, [[2.0, 4.0]])
    np.testing.assert_allclose(loaded[0].estimator.intercept_, [5.0])

    # Raw weights for StandardScaler are unnormalized before writing.
    np.testing.assert_allclose(loaded[1].estimator.coef_, [[1.0, 1.0]])
    np.testing.assert_allclose(loaded[1].estimator.intercept_, [2.0])
