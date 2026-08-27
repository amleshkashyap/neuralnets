import nni

# returns parameters for the trial
params = nni.get_next_parameter()

x = params['x']
y = params['y']
z = params['z']

metric = x + y + z

# return final trial metric
nni.report_final_result(metric)