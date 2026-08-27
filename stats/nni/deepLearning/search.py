import time
from pathlib import Path
from nni.experiment import Experiment

searchSpace = {
    "optimizer": {"_type": "choice", "_value": ['adam', 'sgd', 'adamax']},
    "gruHiddenSize": {"_type": "choice", "_value": [8, 12, 16, 24, 32]},
    "learningRate": {"_type": "choice", "_value": [0.001, 0.005, 0.01]}
}

if __name__ == "__main__":
    maxTrials = 30
    search = Experiment('local')
    search.config.experiment_name = 'GRU Search'
    search.config.trial_concurrency = 7
    search.config.max_trial_number = maxTrials
    search.config.search_space = searchSpace
    search.config.trial_command = 'python main.py'
    search.config.trial_code_directory = Path(__file__).parent

    search.config.tuner.name = 'Evolution'
    search.config.tuner.class_args['optimize_mode'] = 'minimize'
    search.config.tuner.class_args['population_size'] = 8

    search.start(8082)

    executed = 0
    while True:
        trials = search.export_data()
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