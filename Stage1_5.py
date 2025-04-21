import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import pickle
from tqdm.auto import tqdm
import mne

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.inspection import permutation_importance

# Try Bayesian optimization; fallback to GridSearchCV
try:
    from skopt import BayesSearchCV as SearchCV
    search_name = 'Bayesian Search CV'
    search_param_name = 'search_spaces'
    search_kwargs = {'n_iter': 32}
except ImportError:
    from sklearn.model_selection import GridSearchCV as SearchCV
    search_name = 'Grid Search CV'
    search_param_name = 'param_grid'
    search_kwargs = {}

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier

# ----------------------------------------
# 1. Initialization and Settings
# ----------------------------------------
print("🤖 [Initializing search strategy]")
print(f"✨ Using {search_name} ✨")
warnings.filterwarnings('ignore')
print("🎉 Libraries loaded and warnings suppressed! 🎉")

# Define frequency bands
freq_bands = {
    'delta (1.0, 4.0)': (1.0, 4.0),
    'theta-low (4.0, 5.1)': (4.0, 5.1),
    'theta-med (5.1, 6.5)': (5.1, 6.5),
    'theta-high (6.5, 8.0)': (6.5, 8.0),
    'alpha-low (8.0, 10.5)': (8.0, 10.5),
    'alpha-high (10.5, 13.4)': (10.5, 13.4),
    'beta-low (13.4, 17.0)': (13.4, 17.0),
    'beta-med (17.0, 21.7)': (17.0, 21.7),
    'beta-high (21.7, 27.6)': (21.7, 27.6),
    'beta-high/gamma-low (27.6, 35.2)': (27.6, 35.2),
    'gamma-high (35.2, 44.8)': (35.2, 44.8),
}
print("🧮 [Defining frequency bands]")
print(f"✅ Defined {len(freq_bands)} frequency bands: {list(freq_bands.keys())} 🎶")

def label_freq(f):
    for name, (lo, hi) in freq_bands.items():
        if lo <= f < hi:
            return name
    return 'out-of-range'

# ----------------------------------------
# 2. Data Loading & Preprocessing
# ----------------------------------------
print("🚀 Starting data load and preprocessing")
with open('biomarker_data.pkl', 'rb') as f:
    data = pickle.load(f)
print("📥 Data loaded from biomarker_data.pkl")

eeg_xda = data['electrode_space']['eeg_xdata']
all_labels = eeg_xda.coords['condition'].values
print(f"    🔍 Raw data shape: {eeg_xda.shape}")
print(f"    🎭 Conditions: {np.unique(all_labels)}")

# Filter to binary classes
binary = ['ECR', 'SST']
mask = np.isin(all_labels, binary)
X = eeg_xda.values.reshape(len(all_labels), -1)[mask]
y = LabelEncoder().fit_transform(all_labels[mask])
print(f"2️⃣ Filtered to {binary}: {X.shape[0]} samples (ECR={sum(y==0)}, SST={sum(y==1)})")

# Scale features
X = StandardScaler().fit_transform(X)
print("⚖️ Scaling complete! 🥳")
print(f"    🧲 Electrodes: {eeg_xda.shape[1]}, Frequencies: {eeg_xda.shape[2]}, Biomarkers: {eeg_xda.shape[3]}")

# ----------------------------------------
# 3. Model Definitions
# ----------------------------------------
print("🔧 -----Setting up models and cross-validation-----")
models = {
    'LogisticRegression': (
        LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42),
        {'clf__C': [1e-6, 1e-3, 1e-1, 1, 10], 'clf__penalty': ['l1','l2']}
    ),
    'DecisionTree': (
        DecisionTreeClassifier(class_weight='balanced', random_state=42),
        {'clf__max_depth': list(range(1,21)), 'clf__min_samples_split': list(range(2,21))}
    ),
    'GaussianNB': (
        GaussianNB(),
        {'clf__var_smoothing': [1e-12, 1e-10, 1e-8, 1e-6]}
    ),
    'SVC_RBF': (
        SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42),
        {'clf__C': [1e-3, 1e-1, 1, 10], 'clf__gamma': [1e-4, 1e-2, 1e-1]}
    ),
    'HistGradientBoosting': (
        HistGradientBoostingClassifier(random_state=42),
        {'clf__max_iter': [50,100,200], 'clf__learning_rate': [1e-3,1e-2,1e-1], 'clf__max_depth': [1,3,5]}
    ),
    'ElasticNet': (
        SGDClassifier(penalty='elasticnet', class_weight='balanced', random_state=42),
        {'clf__alpha': [1e-6,1e-4,1e-2,1e-1], 'clf__l1_ratio': [0.0,0.25,0.5,0.75,1.0]}
    ),
    'RandomForest': (
        RandomForestClassifier(class_weight='balanced', random_state=42),
        {'clf__n_estimators': [50,100,200], 'clf__max_depth': [None,5,10]}
    ),
    'XGBoost': (
        XGBClassifier(eval_metric='logloss', random_state=42),
        {'clf__n_estimators': [50,100,200], 'clf__learning_rate': [1e-3,1e-2,1e-1], 'clf__max_depth': [1,3,5]}
    ),
}
print(f"🛠️ {len(models)} models ready for evaluation.")

# ----------------------------------------
# 4. Visualization Setup
# ----------------------------------------
print("🌈 -----Defining visualization function-----")
# heatmap and topomap plotting defined later

# ----------------------------------------
# 5. Main Pipeline
# ----------------------------------------
print("🚀 [Running full pipeline in __main__]")
print("🚦 Beginning nested cross-validation: Outer 5-fold, Inner 5-fold")
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}
feature_imps = {}
n_models = len(models)

for name, (model, param_grid) in tqdm(models.items(), total=n_models, desc="Models"):
    print(f"\n🚦 Evaluating model: {name}")
    fold_metrics = {'accuracy':[], 'precision':[], 'recall':[], 'f1':[], 'auc':[]}
    feat_accum = np.zeros(X.shape[1])
    mean_fpr = np.linspace(0,1,100)
    tprs = []

    for fold, (tr, te) in enumerate(tqdm(outer_cv.split(X,y), total=5, desc=f"{name} folds"),1):
        print(f"    🔄 Fold {fold}/5")
        pipe = Pipeline([('imputer', SimpleImputer()), ('clf', model)])
        search = SearchCV(pipe,
                          **{search_param_name: param_grid},
                          cv=inner_cv,
                          scoring='accuracy',
                          n_jobs=-1,
                          random_state=42,
                          **search_kwargs)
        search.fit(X[tr], y[tr])
        best = search.best_estimator_
        print(f"    ✅ Best params: {search.best_params_}")

        y_pred = best.predict(X[te])
        y_proba = best.predict_proba(X[te])[:,1]
        # scores
        fold_metrics['accuracy'].append(accuracy_score(y[te], y_pred))
        fold_metrics['precision'].append(precision_score(y[te], y_pred))
        fold_metrics['recall'].append(recall_score(y[te], y_pred))
        fold_metrics['f1'].append(f1_score(y[te], y_pred))
        fpr,tpr,_ = roc_curve(y[te], y_proba)
        interp = np.interp(mean_fpr, fpr, tpr); interp[0]=0.0
        tprs.append(interp)
        fold_metrics['auc'].append(auc(fpr,tpr))

        # permutation importance
        imp = permutation_importance(best, X[te], y[te], n_repeats=10, random_state=42)
        feat_accum += imp.importances_mean

    # aggregate
    feature_imps[name] = feat_accum / outer_cv.get_n_splits()
    results[name] = {
        'mean_fpr': mean_fpr,
        'mean_tpr': np.mean(tprs,axis=0),
        'f1': np.mean(fold_metrics['f1']),
        'auc': np.mean(fold_metrics['auc']),
    }
    print(f"✅ Completed {name}. Averaged importances computed.")

# ----------------------------------------
# 6. Results & Visualizations
# ----------------------------------------
print("\n📊 Plotting ROC curves per model...")
plt.figure(figsize=(10,8))
for name,res in results.items():
    plt.plot(res['mean_fpr'],res['mean_tpr'],label=f"{name} (AUC={res['auc']:.2f})")
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curves per Model')
plt.legend(loc='lower right'); plt.show()

# heatmap & topomap
print("\n📊 Plotting feature importance heatmaps & topomaps...")

# prepare for topomap
montage = mne.channels.make_standard_montage('biosemi128')
info = mne.create_info(ch_names=montage.ch_names, sfreq=256., ch_types='eeg')
info.set_montage(montage)

for name, imp in feature_imps.items():
    print(f"\n🔎 {name} importance heatmap & topomap")
    arr = imp.reshape(eeg_xda.shape[1], eeg_xda.shape[2], eeg_xda.shape[3])
    # heatmap (freq x biomarker)
    mat = arr.mean(axis=0)
    plt.figure(figsize=(6,4))
    plt.imshow(mat, aspect='auto')
    plt.xticks(range(eeg_xda.shape[3]), eeg_xda.coords['biomarker'].values, rotation=45)
    plt.yticks(range(eeg_xda.shape[2]), [label_freq(f) for f in eeg_xda.coords['frequency'].values])
    plt.title(f"Feature Importance Heatmap: {name}")
    plt.colorbar(); plt.tight_layout(); plt.show()
    # topomap
    elec_imp = arr.mean(axis=(1,2))
    fig,ax = plt.subplots(figsize=(5,4))
    mne.viz.plot_topomap(elec_imp, info, axes=ax, show=False)
    ax.set_title(f"Electrode Topomap: {name}")
    plt.show()

# ----------------------------------------
# 7. Benchmark Model
# ----------------------------------------
best = max(results, key=lambda k: results[k]['f1'])
print(f"\n🏆 Best model by F1: {best} (F1={results[best]['f1']:.3f})")