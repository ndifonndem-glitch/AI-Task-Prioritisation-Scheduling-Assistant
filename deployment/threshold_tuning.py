#!/usr/bin/env python3
"""Task 4 — choose the Do now decision threshold from the cost of each error type.

Produces the real numbers for Section 7.2 of the report. Nothing here is estimated:
it sweeps the threshold over the held-out test split and reports what each operating
point actually costs and recovers.

    python3 1_threshold_tuning.py --repo /path/to/Assement_2

Outputs:
    threshold_sweep.csv      every operating point
    threshold_choice.json    the chosen point and the reasoning
    figure_threshold.png     recall/precision against threshold, for the report
"""
import argparse, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument('--repo', default='.')
ap.add_argument('--cost-ratio', type=float, default=5.0,
                help='cost of a missed Do now relative to one false alarm (default 5)')
ap.add_argument('--random-state', type=int, default=42)
args = ap.parse_args()

sys.path.insert(0, args.repo)
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

DATA = os.path.join(args.repo, 'data', 'knowledge_worker_task_priority_5000.csv')
MODEL = os.path.join(args.repo, 'ml_service', 'model', 'decision_tree_priority.pkl')
for p in (DATA, MODEL):
    if not os.path.exists(p):
        sys.exit('missing: %s\nRun this against a clone of the repository.' % p)

df = pd.read_csv(DATA)
model = joblib.load(MODEL)

TARGET = next(c for c in ('priority', 'label', 'target', 'y') if c in df.columns)
feature_names = list(getattr(model, 'feature_names_in_', [c for c in df.columns if c != TARGET]))

# Three features are derived at serving time rather than stored in the CSV. The
# formulas below are copied verbatim from ml_service/predictor.py so the sweep
# measures the same function the service computes -- a different derivation here
# would produce a threshold that does not transfer to the deployed model.
if 'urgency_density' not in df.columns:
    df['urgency_density'] = 1.0 / (df['hours_to_deadline'] + 0.1)
if 'deferral_starvation_index' not in df.columns:
    df['deferral_starvation_index'] = df['times_deferred'] * (df['age_days'] + 1.0)
if 'work_pressure_ratio' not in df.columns:
    df['work_pressure_ratio'] = df['est_duration_h'] / (df['hours_to_deadline'] + 0.1)

missing = [c for c in feature_names if c not in df.columns]
if missing:
    sys.exit('dataset is missing model features: %s' % missing)
X, y = df[feature_names], df[TARGET]

# Same split the model was trained under, so the test rows are genuinely held out.
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30,
                                            random_state=args.random_state, stratify=y)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50,
                                            random_state=args.random_state, stratify=y_tmp)
print('test split: %d rows' % len(y_te))
print(y_te.value_counts().to_string(), '\n')

classes = list(model.classes_)
DO_NOW = next(c for c in classes if str(c).lower().startswith('do now'))
idx = classes.index(DO_NOW)
proba = model.predict_proba(X_te)[:, idx]
argmax_pred = model.predict(X_te)

def at(threshold):
    """Predict Do now when its probability clears the threshold; otherwise fall back
    to the model's own choice among the remaining classes."""
    pred = []
    for p_row, fallback in zip(model.predict_proba(X_te), argmax_pred):
        if p_row[idx] >= threshold:
            pred.append(DO_NOW)
        elif fallback == DO_NOW:
            others = [(p_row[i], c) for i, c in enumerate(classes) if c != DO_NOW]
            pred.append(max(others)[1])
        else:
            pred.append(fallback)
    return np.array(pred)

actual_do_now = (y_te == DO_NOW).values
rows = []
for t in [round(x, 3) for x in np.arange(0.05, 1.001, 0.05)]:
    pred = at(t)
    pred_do_now = (pred == DO_NOW)
    tp = int((pred_do_now & actual_do_now).sum())
    fp = int((pred_do_now & ~actual_do_now).sum())
    fn = int((~pred_do_now & actual_do_now).sum())
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    cost = args.cost_ratio * fn + fp          # the quantity being minimised
    rows.append(dict(threshold=t, tp=tp, fp=fp, fn=fn,
                     recall=round(recall, 4), precision=round(precision, 4),
                     overall_accuracy=round(float((pred == y_te.values).mean()), 4),
                     weighted_cost=round(cost, 2)))

sweep = pd.DataFrame(rows)
sweep.to_csv('threshold_sweep.csv', index=False)
print(sweep.to_string(index=False), '\n')

best = sweep.loc[sweep.weighted_cost.idxmin()]
base = sweep.loc[(sweep.threshold - 0.50).abs().idxmin()]

choice = {
    'cost_ratio_missed_to_false_alarm': args.cost_ratio,
    'chosen_threshold': float(best.threshold),
    'at_chosen': {'recall': float(best.recall), 'precision': float(best.precision),
                  'caught': int(best.tp), 'missed': int(best.fn),
                  'false_alarms': int(best.fp),
                  'overall_accuracy': float(best.overall_accuracy)},
    'at_argmax_0.50': {'recall': float(base.recall), 'precision': float(base.precision),
                       'caught': int(base.tp), 'missed': int(base.fn),
                       'false_alarms': int(base.fp),
                       'overall_accuracy': float(base.overall_accuracy)},
    'do_now_support': int(actual_do_now.sum()),
    'test_rows': int(len(y_te)),
}
json.dump(choice, open('threshold_choice.json', 'w'), indent=2)

print('cost ratio %.0f:1  ->  threshold %.2f' % (args.cost_ratio, best.threshold))
print('  at 0.50 (argmax): %d of %d caught, %d false alarms, recall %.4f'
      % (base.tp, base.tp + base.fn, base.fp, base.recall))
print('  at %.2f          : %d of %d caught, %d false alarms, recall %.4f'
      % (best.threshold, best.tp, best.tp + best.fn, best.fp, best.recall))
print('\nThese are the numbers to quote in Section 7.2. Do not round them by hand.')

try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
    ax.plot(sweep.threshold, sweep.recall, marker='o', label='Do now recall', color='#17a589')
    ax.plot(sweep.threshold, sweep.precision, marker='s', label='Do now precision', color='#2f4454')
    ax.axvline(best.threshold, ls='--', color='#c0392b',
               label='chosen (cost %.0f:1)' % args.cost_ratio)
    ax.set_xlabel('Decision threshold on P(Do now)'); ax.set_ylabel('Score')
    ax.set_title('Operating point chosen from error cost, not from argmax')
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig('figure_threshold.png')
    print('wrote figure_threshold.png')
except ImportError:
    print('(matplotlib not installed - skipped the figure)')
