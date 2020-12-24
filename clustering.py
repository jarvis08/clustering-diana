import sys
import numpy as np
from scipy.spatial import distance_matrix


def load_dataset(path):
    with open(path, 'r') as f:
        tracker = f.readlines()
        n_samples = len(tracker)
        data = [[0.0, 0.0] for _ in range(n_samples)]
        for i in range(n_samples):
            d = tracker[i].split()
            data[i][0] = float(d[1])
            data[i][1] = float(d[2].replace("\n", ''))
    return data, n_samples


def save_result(path, ctr):
    with open(path, 'w') as f:
        for i in range(len(ctr)):
            f.write(str(ctr[i]) + "\n")


def diana(data_path, k):
    print(">> Start!!")
    objects, n_objects = load_dataset(data_path)
    clusters = [list(range(n_objects))]
    dists = distance_matrix(objects, objects, p=2)

    while len(clusters) < k:
        print("Length of clusters = {}".format(len(clusters)))
        diameters = [np.max(dists[ctr][:, ctr]) for ctr in clusters]
        max_dtr = np.argmax(diameters)

        avg_dists_in_ctr = np.mean(dists[clusters[max_dtr]][:, clusters[max_dtr]], axis=1)
        seed_idx = np.argmax(avg_dists_in_ctr)

        splinter_group = [clusters[max_dtr][seed_idx]]
        non_splinter = clusters[max_dtr].copy()
        non_splinter.pop(seed_idx)

        while True:
            in_dist = np.mean(dists[non_splinter][:, splinter_group], axis=1)
            out_dist = dists[non_splinter][:, non_splinter]
            out_dist = np.mean(out_dist, axis=1) - 1 / len(non_splinter) # to exclude i to i distance(=1)
            d_h = out_dist - in_dist
            idx = np.argmax(d_h)
            if d_h[idx] > 0:
                splinter_group.append(non_splinter[idx])
                non_splinter.pop(idx)
            else:
                break
        del clusters[max_dtr]
        clusters.append(non_splinter)
        clusters.append(splinter_group)
    print("Length of clusters = {}".format(len(clusters)))
    print("Save results")
    for i in range(k):
        save_result(data_path.replace(".txt", '') + '_cluster_{}.txt'.format(i), clusters[i])
    print(">> Done.")


if __name__ == "__main__":
    argv = sys.argv
    if len(argv) < 3:
        print("Not enough arguments given.")
        exit()
    INPUT_FILENAME = argv[1]
    N_CLUSTERS = argv[2]
    print("Run decision tree algorithm with given arguments.")
    print("Input filename : {}".format(INPUT_FILENAME))
    print("Number of Clusters : {}".format(N_CLUSTERS))
    diana(INPUT_FILENAME, int(N_CLUSTERS))

