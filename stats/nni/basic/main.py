import time
from pathlib import Path
from nni.experiment import Experiment

searchSpace = {
    "x": {"_type": "choice", "_value": [-10, -5, 0, 5, 10]},
    "y": {"_type": "choice", "_value": [-10, -5, 0, 5, 10]},
    "z": {"_type": "choice", "_value": [-10, -5, 0, 5, 10]}
}

if __name__ == "__main__":
    search = Experiment('local')
    search.config.experiment_name = 'Basic Search'
    search.config.trial_concurrency = 4
    search.config.max_trial_number = 50
    search.config.search_space = searchSpace
    # use the relevant command - python3 or python depending on environment
    search.config.trial_command = 'python trial.py'
    search.config.trial_code_directory = Path(__file__).parent

    # configure naive evolution tuner
    search.config.tuner.name = 'Evolution'
    search.config.tuner.class_args['optimize_mode'] = 'minimize'
    search.config.tuner.class_args['population_size'] = 8

    search.start(8082)

    while True:
        if search.get_status() == 'DONE':
            trials = search.export_data()
            print(trials)
            bestTrial = min(trials, key = lambda t: t.value)
            print(f'Best Trial Params: {bestTrial.parameter}')
            input("Press Key To Exit....")
            break
        time.sleep(10)