import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from attributes import Attributes
from transforms import add_labels, extract_bt_df, Preprocessing, balance, expected_value, backtest
from visuals import feature_ranking
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier


if __name__ == '__main__':
    # Load and label the data
    df = pd.read_csv('../data/city.csv', low_memory=False)
    df['assessor_id'] = df['assessor_id'].str[1:]
    df = add_labels(df)

    # Clean, drop, and engineer features. Impute missing values.
    clean = Preprocessing()
    df = clean.transform(df)

    # Extract subset for backtesting map
    # data = extract_bt_df(df)
    # data = df

    # Scale numerical features
    cols_to_scale = Attributes().get_num_attribs()
    scaler = RobustScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    y = df.pop('labels')
    X = df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    X_train_res, y_train_res, idx_res = balance(X_train, y_train, method='downsample')

    # Train and get feature importances
    model = RandomForestClassifier(
            random_state = 42,
            bootstrap=True,
            criterion='entropy',
            max_depth=5,
            max_features='auto',
            min_samples_leaf=20,
            min_samples_split=10,
            n_estimators=200,
            n_jobs=-1,
            class_weight='balanced_subsample',
            oob_score=False
    )

    model.fit(X_train, y_train)
    print("Model is fit and ready to predict.")

    # Report feature importances
    n = 10
    importances = model.feature_importances_[:n]
    indices = np.argsort(importances)[::-1]
    features = list(X.columns[indices])
    print("\n Feature Ranking:")
    for f in range(n):
        print("%d. %s (%f)" % (f + 1, features[f], importances[indices[f]]))

    # # Plot the feature importances
    # feature_ranking(importances, indices, features)

    # Score the final model
    y_pred = model.predict(X_test)
    y_probs = (model.predict_proba(X_test).T)[1]
    accuracy = model.score(X_test, y_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("Accuracy: {}".format(accuracy))
    print("Confusion Matrix:")
    print("TP: {}".format(tp))
    print("FP: {}".format(fp))
    print("FN: {}".format(fn))
    print("TN: {}".format(tn))
    print("Final results:")
    print(classification_report(y_test, y_pred))

    # Pickle and save final model
    # with open('model.pkl', 'wb') as f:
    #     pickle.dump(model, f)

    # # Make backtesting predictions for map
    # backtest('model.pkl', '../data/2016_backtest.csv')

    # Get # TP and FP by probabilities, for expected value calucation
    numTP, numFP = expected_value(y_test, y_pred, y_probs, num_jobs=400)
