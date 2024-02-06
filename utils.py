from nilearn import datasets, input_data, plotting, connectome
import nibabel as nib
import numpy as np

def load_fmri_data():
    adhd_dataset = datasets.fetch_adhd(n_subjects=1)
    fmri_img = nib.load(adhd_dataset.func[0])

    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    masker = input_data.NiftiLabelsMasker(labels_img=atlas.maps, standardize=True)

    roi_time_series = masker.fit_transform(fmri_img)
    return roi_time_series, atlas.labels
    # correlation_measure = connectome.ConnectivityMeasure(kind='correlation')

def normalize(data):
    roi_time_series_normalized = np.zeros_like(data)

    for i in range(data.shape[0]):
        row = data[i, :]
        min_val = np.min(row)
        max_val = np.max(row)

        # Avoid division by zero
        if max_val != min_val:
            normalized_row = (row - min_val) / (max_val - min_val)
            roi_time_series_normalized[i, :] = normalized_row

    return roi_time_series_normalized


def plot(normalized_transition_matrix, labels, num_regions=10):
    import networkx as nx
    import matplotlib.pyplot as plt

    G = nx.DiGraph()


    G.add_nodes_from(range(num_regions))


    for i in range(num_regions):
        for j in range(num_regions):
            if i != j:
                transition_probability = normalized_transition_matrix[i, j]
                G.add_edge(i, j, weight=transition_probability)


    pos = nx.circular_layout(G) 
    labels = {i: labels[i] for i in range(10)}

    edge_labels = {(i, j): f'{normalized_transition_matrix[i, j]:.2f}' for i in range(num_regions) for j in range(num_regions) if i != j and float(normalized_transition_matrix[i,j]) > 0.0}
    print(edge_labels)

    nx.draw(G, pos, with_labels=True, labels=labels, font_size=6, node_size=3000, font_color='black', node_color='skyblue')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    plt.title('Brain Region Transition Probabilities')