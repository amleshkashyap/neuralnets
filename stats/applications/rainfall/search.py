from nni import Experiment
import time
from pathlib import Path

searchSpace = {
    'tclNum': {
        '_type': 'choice',
        '_value': [1, 2, 3]
    },
    'tclChannelSize': {
        '_type': 'choice',
        '_value': [8, 16, 24, 32]
    },
    'kernelSize': {
        '_type': 'choice',
        '_value': [3, 5, 7]
    },
    'dropout': {
        '_type': 'choice',
        '_value': [0, 0.1, 0.2, 0.4]
    },
    'slices': {
        '_type': 'choice',
        '_value': [1, 2]
    },
    'useBias': {
        '_type': 'choice',
        '_value': [True, False]
    },
    'lr': {
        '_type': 'choice',
        '_value': [0.01, 0.005, 0.0001]
    }
}

if __name__ == "__main__":
    maxTrials = 30
    search = Experiment('local')
    search.config.experiment_name = 'TCN Search'
    search.config.trial_concurrency = 7
    search.config.max_trial_number = maxTrials
    search.config.search_space = searchSpace
    search.config.trial_command = 'python main.py'
    search.config.trial_code_directory = Path(__file__).parent

    search.config.tuner.name = 'Evolution'
    search.config.tuner.class_args['optimize_mode'] = 'minimize'
    search.config.tuner.class_args['population_size'] = 32

    search.start(8082)

    executed = 0
    try:
        while True:
            trials = search.export_data()
            print(trials)
            if executed != len(trials):
                executed = len(trials)
                print(f'\nTrials: {executed} / {maxTrials}')
            if search.get_status() == 'DONE':
                bestTrial = min(trials, key = lambda t: t.value)
                print(f'Best Trial Params: {bestTrial.parameter}')
                input("Press Key To Exit...")
                break
            print(".", end = "")
            time.sleep(10)
    except Exception as ex:
        print("Error: ", ex)
        input("Encountered An Exception, Press Key To Exit...")