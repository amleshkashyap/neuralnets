import time
from nni.experiment import Experiment

fastChoices = {
    '_type': 'choice',
    '_value': [3, 5, 7, 9]
}
slowChoices = {
    '_type': 'choice',
    '_value': [14, 20, 40]
}
lengthChoices = {
    '_type': 'choice',
    '_value': [5, 10, 20]
}

indicatorChoices = [
    {
        '_name': 'ao',
        'fast': fastChoices,
        'slow': slowChoices
    },
    {
        '_name': 'apo',
        'fast': fastChoices,
        'slow': slowChoices
    },
    {
        '_name': 'tsi',
        'fast': fastChoices,
        'slow': slowChoices
    },
    {
        '_name': 'cci',
        'length': lengthChoices
    },
    {
        '_name': 'cmo',
        'length': lengthChoices
    },
    {
        '_name': 'mom',
        'length': lengthChoices
    },
    {
        '_name': 'rsi',
        'length': lengthChoices
    }
]

searchSpace = {
    'lr': {
        '_type': 'choice',
        '_value': [0.01, 0.005, 0.001, 0.0005]
    },
    'rnnType': {
        '_type': 'choice',
        '_value': ['rnn', 'gru']
    },
    'rnnHiddenSize': {
        '_type': 'choice',
        '_value': [8, 16, 24]
    },
    'indicatorHiddenSize': {
        '_type': 'choice',
        '_value': [1, 2, 4]
    },
    'decisionSize': {
        '_type': 'choice',
        '_value': [2, 4, 8, 16]
    },
    'ind1': {
        '_type': 'choice',
        '_value': indicatorChoices
    },
    'ind2': {
        '_type': 'choice',
        '_value': indicatorChoices
    }
}

if __name__ == "__main__":
    maxTrials = 300
    search = Experiment('local')
    search.config.experiment_name = 'TCN Search'
    search.config.trial_concurrency = 2
    search.config.max_trial_number = maxTrials
    search.config.search_space = searchSpace
    search.config.trial_command = 'python main.py'
    search.config.trial_code_directory = '.'

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