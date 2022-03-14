import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn import ensemble
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

def plot_regression_deviance(reg, params, X_test_norm, y_test):

    test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
    for i, y_pred in enumerate(reg.staged_predict(X_test_norm)):
        test_score[i] = reg.loss_(y_test, y_pred)

    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.title("Deviance")
    plt.plot(
        np.arange(params["n_estimators"]) + 1,
        reg.train_score_,
        "b-",
        label="Training Set Deviance",
    )
    plt.plot(
        np.arange(params["n_estimators"]) + 1, test_score, "r-", label="Test Set Deviance"
    )
    plt.legend(loc="upper right")
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Deviance")
    fig.tight_layout()
    plt.show(block=False)

def plot_feature_importance(reg, feature_names, X_test_norm, y_test):

    feature_importance = reg.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(pos, feature_importance[sorted_idx], align="center")
    plt.yticks(pos, np.array(feature_names)[sorted_idx])
    plt.title("Feature Importance (MDI)")

    result = permutation_importance(
        reg, X_test_norm, y_test, n_repeats=10, random_state=42, n_jobs=2
    )
    sorted_idx = result.importances_mean.argsort()
    plt.subplot(1, 2, 2)
    plt.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        labels=np.array(feature_names)[sorted_idx],
    )
    plt.title("Permutation Importance (test set)")
    fig.tight_layout()
    plt.show(block=False)

def plot_weights_accuracy_scatter(reg, X_train_norm, y_train, X_test_norm, y_test):

    fig, axes = plt.subplots(nrows=1, ncols=2)
    ax = axes[0]
    y_pred = reg.predict(X_train_norm)
    y_true = y_train
    ax.scatter(y_true, y_pred)
    ax.plot([-3,2], [-3,2], 'k')
    ax.set_title("Train")

    ax = axes[1]
    y_pred = reg.predict(X_test_norm)
    y_true = y_test
    ax.scatter(y_true, y_pred)
    ax.plot([-3,2], [-3,2], 'k')
    ax.set_title("Test")
    plt.show(block=False)


def gb_regression(params, X_train, y_train, X_test, y_test, feature_names, show=False):
    # Normalize the features 
    norm = MinMaxScaler().fit(X_train)
    X_train_norm = norm.transform(X_train)
    X_test_norm = norm.transform(X_test)


    reg = ensemble.GradientBoostingRegressor(**params)
    reg.fit(X_train_norm, y_train)
    error = mean_squared_error(y_test, reg.predict(X_test_norm))
    print(f"Regression mean squre error: {error}")
    if show:
        plot_regression_deviance(reg, params, X_test_norm, y_test)
        plot_feature_importance(reg, feature_names, X_test_norm, y_test)
        plot_weights_accuracy_scatter(reg, X_train_norm, y_train, X_test_norm, y_test)
    return reg, norm



def split_mode_traintest(wp_mode, train_ratio):
    N_s = len(wp_mode)
    wp_train = wp_mode.iloc[:int(train_ratio*N_s)]
    wp_test = wp_mode.iloc[int(train_ratio*N_s):]
    return wp_train, wp_test


def get_regressors_for_each_mode(N_modes, params, wp, feature_names, train_ratio=1, show=False):
    regs = []
    norms = []
    for mode in range(N_modes):
        print(f"---regression mode:{mode}---")
        wp_mode = wp[wp['mode'] == mode]
        wp_train, wp_test = split_mode_traintest(wp_mode.sample(frac=1), train_ratio)
        print(wp_train)

        # Prepare train and test
        X_train = wp_train[feature_names]
        y_train = wp_train['w']
        X_test = wp_test[feature_names]
        y_test = wp_test['w']

        reg, norm = gb_regression(params, X_train, y_train, X_test, y_test, feature_names, show)
        regs.append(reg)
        norms.append(norm)
    return regs, norms
