import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from sklearn.model_selection import GridSearchCV
from sklearn import metrics


def one_hot_encode(feature, prefix):
    return pd.get_dummies(data=feature,
                          prefix=prefix,
                          drop_first=True,
                          dummy_na=True).astype('int')


def GridSearchCVWrapper(model, param_grid, X, y, n_jobs=-1, cv=10):
    clf_cv = GridSearchCV(model, param_grid=param_grid,
                          n_jobs=n_jobs, cv=cv,
                          scoring='recall_weighted')
    clf = clf_cv.fit(X, y)
    best_params = clf.best_params_
    best_score = round(clf.best_score_, 3)
    print('Best Params: {}\n'
          'Best Score: {}'.format(best_params, best_score))
    return best_params, best_score




def precision_recall_thershold(probas, y_test):
    t_precision_diab = []
    t_recall_diab = []
    t_precision_nodiab = []
    t_recall_nodiab = []

    thresholds = np.arange(0, 1, 0.01)
    for thresh in thresholds:
        y_pred = np.where(probas > thresh, 1, 0)
        precision, recall, _, _ = metrics.precision_recall_fscore_support(y_test, y_pred)
        # print("PRECISION: ", precision) 
        t_precision_diab.append(precision[1])
        t_recall_diab.append(recall[1])
        t_precision_nodiab.append(precision[0])
        t_recall_nodiab.append(recall[0])

    return t_precision_nodiab, t_precision_diab, t_recall_nodiab, t_recall_diab


def bootstrap_model(model, X, y, X_test, y_test, n_bootstrap, thresh):
    total_recall = []
    total_precision = []
    total_fscore = []
    total_accuracy = []
    total_fpr_tpr = []
    results = []
    size = X.shape[0]

    for _ in range(n_bootstrap):
        boot_ind = np.random.randint(size, size=size)
        X_boot = X.loc[boot_ind]
        y_boot = y.loc[boot_ind]

        clf = model.fit(X_boot, y_boot)
        y_pred = clf.predict_proba(X_test)[:, 1]
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            y_test, np.where(y_pred > thresh, 1, 0))
        accuracy = metrics.accuracy_score(y_test, np.where(y_pred > thresh, 1, 0))

        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred, pos_label=1)
        fpr_tpr = (fpr, tpr)
        total_fpr_tpr.append(fpr_tpr)
        total_recall.append(recall[1])
        total_precision.append(precision[1])
        total_fscore.append(fscore[1])
        total_accuracy.append(accuracy)
        results.append({'y_test': y_test, 'y_pred': np.where(y_pred > thresh, 1, 0)})

    results_dict = dict(recall=total_recall,
                        precision=total_precision,
                        fscore=total_fscore,
                        accuracy=total_accuracy,
                        fpr_tpr=total_fpr_tpr,
                        results=results)

    return results_dict


def roc_interp(fpr_tpr):
    linsp = np.linspace(0, 1, 100)
    n_boot = len(fpr_tpr)
    ys = []
    for n in fpr_tpr:
        x, y = n
        interp = np.interp(linsp, x, y)
        ys.append(interp)
    return ys


def plot_recall_vs_decision_boundary(
        t_recall_diab,
        t_recall_nodiab,
        filename='./images/Recall_score.png'):

    plt.figure(figsize=(10,7))
    plt.plot(np.arange(0, 1, 0.01), t_recall_diab,   label='Diabetics')
    plt.plot(np.arange(0, 1, 0.01), t_recall_nodiab, label='Non-Diabetics')
    plt.plot([.5, .5], [0, 1], 'k--')
    plt.plot([.77, .77], [0, 1], 'k--')
    
    plt.ylim([0.0, 1.01])
    plt.xlim([0.0, 1.0])
    plt.legend(loc='upper left', fontsize=14)
    plt.title('Recall vs. Decision Boundary\n'
              'using GradientBoostingClassifier',
              fontsize=14)
    plt.xlabel('Decision Boundary (T)', fontsize=14)
    plt.ylabel('Recall Rate', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig(filename)
    plt.show()


def plot_multi_recall_vs_decision_boundary(
        probas,
        y_test,
        filename='./img/Recall_score_all.png'):

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
    ax1.plot([.5, .5], [0, 1], 'k--')
    ax2.plot([.5, .5], [0, 1], 'k--')
    ax1.set_ylim([0.0, 1.01])
    ax1.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.01])
    ax2.set_xlim([0.0, 1.0])
    ax1.set_xlabel('Decision Boundary (T)', fontsize=14)
    ax1.set_ylabel('Recall Rate', fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax2.set_xlabel('Decision Boundary (T)', fontsize=14)
    ax2.set_ylabel('Recall Rate', fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    for p in probas:
        _, _, t_recall_nodiab, t_recall_diab = \
                precision_recall_thershold(probas[p], y_test)
        ax1.plot(np.arange(0, 1, 0.01), t_recall_diab,   label=p)
        ax1.set_title('Diabetic Class\n'
                      'Recall vs. Decision Boundary',
                      fontsize=14)
        ax2.plot(np.arange(0, 1, 0.01), t_recall_nodiab, label=p)
        ax2.set_title('Non-Diabetic Class\n'
                      'Recall vs. Decision Boundary',
                      fontsize=14)
    ax1.legend(loc='upper left')
    plt.savefig(filename)
    plt.show()

def plot_roc_curves(df_preds, y_test, filename='./img/ROC_curve.png'):
        plt.figure(figsize=(8,8))
        for model in df_preds.columns:
            values = df_preds.loc[:, model].to_numpy().astype(float)
            fpr, tpr, _ = metrics.roc_curve(y_test, values, pos_label=1)
            print('{}\n  AUC: {}'.format(model, round(metrics.auc(fpr, tpr), 3)))
            plt.plot(fpr, tpr, label=model)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.01])
        plt.legend(loc='lower right', fontsize=14)
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('ROC Curve', fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.savefig(filename)
        plt.show()

def plot_bootstrap_roc(m, ci, filename='./img/Bootstrap_ROC_confint.png'):
    x = np.linspace(0,1,100)
    plt.figure(figsize=(8,8))
    plt.plot(x, m, c='blue', label='ROC Mean')
    plt.plot(x, ci[0], c='grey', label='95% CI')
    plt.plot(x, ci[1], c='grey')
    plt.fill_between(x, ci[0], ci[1], color='grey', alpha=0.25)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.legend(loc='lower right', fontsize=14)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Bootstrap ROC Curve', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig(filename)
    plt.show()
