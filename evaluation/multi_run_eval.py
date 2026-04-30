import numpy as np
from models.training_callbacks import default_callbacks
from evaluation.evaluation_metrics import EvaluationMetrics


def multi_run_evaluation(
    build_fn,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    num_runs: int = 3,
    epochs: int = 30,
    batch_size: int = 32,
    description: str = "Model",
):
    accs = []
    f1s = []
    histories = []

    for run in range(num_runs):
        print(f"  {description} run {run + 1}/{num_runs}...", end=" ", flush=True)

        model = build_fn()
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=default_callbacks(),
            verbose=0,
        )

        y_pred = model.predict(X_test, verbose=0)
        metrics = EvaluationMetrics(y_test, y_pred)

        accs.append(metrics.accuracy)
        f1s.append(metrics.f1_macro)
        histories.append(history)

        print(f"acc={metrics.accuracy}, f1={metrics.f1_macro}")

    summary = {
        "name": description,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "accuracy_min": float(np.min(accs)),
        "accuracy_max": float(np.max(accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "all_accs": accs,
        "all_f1s": f1s,
    }

    print(
        f"\n{description}: acc={summary['accuracy_mean']*100}% "
        f"± {summary['accuracy_std']*100}%, "
        f"f1={summary['f1_mean']} ± {summary['f1_std']}"
    )
    return summary, histories
