import numpy as np

class HiddenMarkovModel:
    def __init__(self, num_states):
        self.num_states = num_states
        self.initial_prob = np.ones(num_states) / num_states
        self.transition_matrix = np.ones((num_states, num_states)) / num_states
    
    def fit(self, time_series_data, smoothing=1):
        num_time_points, num_regions = time_series_data.shape
        state_counts = np.zeros((self.num_states, self.num_states))

        for t in range(num_time_points - 1):
            current_state = np.argmax(time_series_data[t])
            next_state = np.argmax(time_series_data[t + 1])
            state_counts[current_state][next_state] += 1

        state_counts += smoothing
        self.transition_matrix = state_counts / np.sum(state_counts, axis=1, keepdims=True)

        