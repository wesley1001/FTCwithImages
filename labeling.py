from tqdm import tqdm
import numpy as np


def triple_barrier_labeling(close, high, low, oppen, params):
    series_length = params['series length']
    upper_ratio = params['upper ratio']
    lower_ratio = params['lower ratio']
    timestemp = params['timestemp']
    step = params['step']
    data_list = []
    for i in tqdm(range(0, len(close) - series_length, step)):
        # print(close)
        array = 0.5*close[i:i+series_length]+0.5*oppen[i:i+series_length]
        high_array = high[i+series_length: i+series_length+timestemp]
        low_array = low[i+series_length: i+series_length+timestemp]
        last_point_open = oppen[i+series_length-1]
        ratios_high = (high_array - last_point_open)/last_point_open
        ratios_low = (low_array - last_point_open)/last_point_open
        indexes_that_reach_high = np.where(ratios_high > upper_ratio)
        indexes_that_reach_low = np.where(ratios_low < lower_ratio)
        if indexes_that_reach_high[0].size == 0:
            reached_high = None
        else:
            reached_high = np.min(indexes_that_reach_high)
        if indexes_that_reach_low[0].size == 0:
            reached_low = None
        else:
            reached_low = np.min(indexes_that_reach_low)

        if reached_high is None:
            if reached_low is None:
                label = 0
            else:
                label = -1
        elif reached_low is None:
            label = 1
        elif reached_low <= reached_high:
            label = -1
        else:
            label = 1
        data_list.append((array, label))
    return data_list
