from hmm import HiddenMarkovModel
from utils import load_fmri_data, normalize, plot


def main():
    roi_time_series, labels = load_fmri_data()
    normalized_roi_time_series = normalize(roi_time_series)

    hmm = HiddenMarkovModel(num_states=48)
    hmm.fit(normalized_roi_time_series)

    plot(hmm.transition_matrix, labels=labels, num_regions=10)

if __name__ == "__main__":
    main()