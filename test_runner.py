import text
import numeric_sequence


def tests():
    return {
        'Chance Classification': {'Numeric': numeric_sequence.classification},#, 'Text': text.classy},
        'xG Regression': {'Numeric': numeric_sequence.regression}#, 'Text': text.chancy}
    }


def run(num_runs=3):
    res = {}
    for task, test_funcs in tests().items():
        for approach, test_func in test_funcs.items():
            if (task, approach) not in res:
                res[(task, approach)] = []
            for r in range(1, num_runs+1):
                res[(task, approach)].append(test_func())
    return res


def average_metric(run_output, target):
    mean = 0
    for (task, approach), runs in run_output.items():
        for output in runs:
            mean += output.metrics[target]
        print(f'{task}, {approach}, {mean/len(runs)}')

