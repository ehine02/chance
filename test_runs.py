import text
import raw_ts


def tests():
    return {
        'Chance Classification': {'Numeric': raw_ts.classy, 'Text': text.classy},
        'xG Regression': {'Numeric': raw_ts.chancy, 'Text': text.chancy}
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
