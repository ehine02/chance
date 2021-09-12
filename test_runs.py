import text
import numeric_sequence


def tests():
    return {
        'Chance Classification': {'Numeric': numeric_sequence.classy, 'Text': text.classy},
        'xG Regression': {'Numeric': numeric_sequence.chancy, 'Text': text.chancy}
    }


def run(num_runs=3):
    res = {}
    for task, test_funcs in tests().items():
        for approach, test_func in test_funcs.items():
            avg = {}
            for r in range(1, num_runs+1):
                avg[r] = (approach, test_func())
            res[task] = avg
    return res
