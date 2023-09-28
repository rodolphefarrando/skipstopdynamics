import numpy as np


def data(path='data/line1.csv'):
    """
    l: length between stations
    s: if a stop is mandatory s = 1, s = 0 otherwise
    v1: speed at a segment before or after a station where a train stops
    v2: full speed
    :return:
    """
    # data = pd.read_csv('data/ligne1.csv', header=None).values()
    csv = np.genfromtxt(path, delimiter=",", encoding="utf-8")
    l = csv[1:, 1]
    s = csv[1:, 2]

    V1 = 30 / 3.6
    V2 = 50 / 3.6

    segments = []
    len_segm = 200

    count = 0

    # stations index all stations
    # skip_stations index stations with s = 0
    stations = []
    skip_stations = []
    for j, l_ in enumerate(l):
        nsegm = int(l_ / len_segm)
        segments += [l_ / nsegm]*nsegm
        stations.append(count)
        if s[j] == 1:
            skip_stations.append(count)

        count += nsegm

    v1 = np.ones((len(segments))) * V2
    v2 = np.ones((len(segments))) * V2
    s1 = np.ones((len(segments))) * 10
    s2 = np.ones((len(segments))) * 10
    w1 = np.zeros((len(segments)))
    w2 = np.zeros((len(segments)))
    for _, stat in enumerate(stations):
        # v1[stat] = V2
        # v1[stat - 1] = V2
        s1[stat] = 20
        w1[stat] = 50 + 20
        if stat == len(w1) - 1:
            w1[0] = 0 #20
        else:
            w1[stat + 1] = 0 # 20
        if stat in skip_stations:
            # v2[stat] = V2
            # v2[stat] = V2
            s2[stat] = 20
            w2[stat] = 50 + 20
            if stat == len(w1) - 1:
                w2[0] = 0 # 20
            else:
                w2[stat + 1] = 0 # 20

    r1 = np.reshape(segments / v1, (len(segments),))
    r2 = np.reshape(segments / v2, (len(segments),))
    t1 = r1 + np.append(w1[1:], w1[0])
    t2 = r2 + np.append(w2[1:], w2[0])

    t2 = np.transpose(np.array([t1, t2]))
    r2 = np.transpose(np.array([r1, r2]))
    s2 = np.transpose(np.array([s1, s2]))

    dist_stat = np.append([0], np.cumsum(segments))
    skip_stations = sorted(list(set(stations) ^ set(skip_stations)))

    return t1, t2, s1, s2, r1, r2, dist_stat, stations, skip_stations


if __name__ == '__main__':
    data()
