import text_sequence
import numeric_sequence


def tests():
    return {
        #'Chance Classification': {'Numeric': numeric_sequence.classification,
        #                          'Text': text_sequence.classification}#,
        'xG Regression': {'Numeric': numeric_sequence.regression,
                          'Text': text_sequence.regression}
    }


def run(num_runs=3, sample_size=5000, epochs=500):
    res = {}
    for task, test_funcs in tests().items():
        for approach, test_func in test_funcs.items():
            if (task, approach) not in res:
                res[(task, approach)] = []
            for r in range(1, num_runs+1):
                res[(task, approach)].append(test_func(sample_size=sample_size, epochs=epochs))
    return res


def average_metric(run_output, target):
    for (task, approach), runs in run_output.items():
        mean = 0
        for output in runs:
            mean += output.metrics.get(target, 0.0)
        print(f'{task}, {approach}, {mean/len(runs)}')

