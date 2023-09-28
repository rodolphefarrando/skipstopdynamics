import time
import numpy as np
import matplotlib as mpl
from .simulator import Simulator
import matplotlib.pyplot as plt
from .data import data
from matplotlib import rc
import scipy as sp

rc('text', usetex=True)


class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return sp.ma.masked_array(np.interp(value, x, y))


def main():
    # Choose the model
    model = 'first'  # or # second

    # get the data from the file 'data/ligne1.csv'
    t1, t2, s1, s2, r1, r2, dist_stat, stations, skip_stations = data()
    n = len(t1)

    # Select the number of trains for which you want to compute the travel time
    m = 22

    # Simulation for the two models; get D and A
    # ONE CYCLE
    sim1 = Simulator(m=m, n=n, s=s1, t=t1, r=r1, horizon=500, I=skip_stations, tt=True, style='linear')
    D1, A1, h1 = sim1.simulation()

    # TWO CYCLES
    sim2 = Simulator(m=m, n=n, s=s2, t=t2, r=r2, horizon=500, I=skip_stations, tt=True,style='linear')
    D2, A2, h2 = sim2.simulation()

    n = int(len(t1) / 2)
    stations = stations[:int(len(stations) / 2)]
    skip_stations = skip_stations[:int(len(skip_stations) / 2)]
    mand_stations = sorted(set(skip_stations) ^ set(stations))
    TT = np.zeros((n, n))
    TT2 = np.zeros((n, n))
    value = -3
    service1 = [stations[2], stations[3], stations[5], stations[11], stations[13], stations[16], stations[22]]
    service2 = [stations[1], stations[7], stations[23]]
    for i in range(TT.shape[0]):
        for j in range(i, TT.shape[1] - 1):
            # TT1 is for the first service and TT2 for the second
            tt1 = 0
            tt2 = 0
            tt3 = 0
            for k in range(i, j + 1):
                tt = A2[k, value + 1] - D2[k, value + 1] \
                    if A2[k, value + 1] >= D2[k, value + 1] else A2[k, value + 1] - D2[k, value + 0]
                tt_ = A1[k, value + 1] - D1[k, value + 1] \
                    if A1[k, value + 1] >= D1[k, value + 1] else A1[k, value + 1] - D1[k, value + 0]
                if k + 1 not in stations or k == j:
                    tt1 += tt
                    tt2 += tt
                elif k + 1 in mand_stations:
                    tt1 += tt + 70
                    tt2 += tt + 70
                elif k + 1 in service1:
                    tt1 += tt
                    tt2 += tt + 70
                elif k + 1 in service2:
                    tt1 += tt + 70
                    tt2 += tt

                if k + 1 not in stations or k == j:
                    tt3 += tt_
                else:
                    tt3 += tt_ + 70

            TT[i, j + 1] = tt3 + h1 / 2
            if i in service1 or (j + 1) in service1:
                TT2[i, j + 1] = tt1 + h2
            elif i in service2 or (j + 1) in service2:
                TT2[i, j + 1] = tt2 + h2
            else:
                TT2[i, j + 1] = np.mean([tt1, tt2]) + h2 / 2

            TT[j + 1, i] = TT[i, j + 1]
            TT2[j + 1, i] = TT2[i, j + 1]

    TT = TT[stations, :][:, stations]
    TT2 = TT2[stations, :][:, stations]

    infeasible_od = ((1, 2), (1, 3), (1, 5), (1, 11), (1, 13), (1, 16), (1, 22),
                     (7, 11), (7, 13), (7, 16), (7, 22),
                     (2, 7), (2, 23),
                     (3, 7), (3, 23),
                     (5, 7), (5, 23),
                     (11, 23), (13, 23), (16, 23), (22, 23))
    DTT = TT - TT2
    vmin = np.min(DTT)
    for inf in infeasible_od:
        # DTT[inf[0], inf[1]] = vmin - 400
        # DTT[inf[1], inf[0]] = vmin - 400
        # TT2[inf[0], inf[1]] += 150
        if inf[1] - inf[0] < 2:
            DTT[inf[0], inf[1]] = TT[inf[0], inf[1]] - (TT2[inf[0], inf[0] - 1]
                                                        + TT2[inf[0] - 1, inf[1]] + h2)
        elif inf[0] < 4:
            DTT[inf[0], inf[1]] = TT[inf[0], inf[1]] - (TT2[inf[0], 4] + TT2[4, inf[1]] + h2)
        elif inf[0] < 6:
            DTT[inf[0], inf[1]] = TT[inf[0], inf[1]] - (TT2[inf[0], 6] + TT2[6, inf[1]] + h2)
        elif inf[0] < 8:
            DTT[inf[0], inf[1]] = TT[inf[0], inf[1]] - (TT2[inf[0], 8] + TT2[8, inf[1]] + h2)
        elif inf[0] < 12:
            DTT[inf[0], inf[1]] = TT[inf[0], inf[1]] - (TT2[inf[0], 12] + TT2[12, inf[1]] + h2)
        elif inf[0] < 14:
            DTT[inf[0], inf[1]] = TT[inf[0], inf[1]] - (TT2[inf[0], 14] + TT2[14, inf[1]] + h2)
        elif inf[0] < 17:
            DTT[inf[0], inf[1]] = TT[inf[0], inf[1]] - (TT2[inf[0], 17] + TT2[17, inf[1]] + h2)
        DTT[inf[1], inf[0]] = DTT[inf[0], inf[1]]

    print(sum(sum(DTT)))
    # print(sum(sum(DTT * OD)))

    fig, ax = plt.subplots(figsize=(12, 7))
    plt.rc('font', family='serif')
    plt.rc('font', size=22)
    vmin = np.min(DTT)
    vmax = np.max(DTT)
    norm = MidpointNormalize(vmin=-360, vmax=360, midpoint=0)
    cax = ax.imshow(DTT, norm=norm, interpolation='nearest',
                    cmap='bwr', aspect='auto')

    SY = [r'\bf{Def.}', '\it{E.Def}', '\it{Neu.}', '\it{Sab.}', r'\bf{Mai.}', '\it{Arg.}', r'\bf{Cdg}',
          '\it{Geo.}', r'\bf{Roo.}', r'\bf{CE}', r'\bf{Con.}', '\it{Tui.}', r'\bf{PR}', '\it{Lou.}',
          r'\bf{Cha.}', r'\bf{Hdv}', '\it{Paul}', r'\bf{Bas.}', r'\bf{GL}', r'\bf{RD}', r'\bf{Nat.}',
          r'\bf{Vin.}', '\it{Man.}', '\it{Ber.}', r'\bf{Ch.v.}']

    SY = [r'Def', '', '', '', r'Mai.', '', r'Cdg', '', r'Roo.', r'CE',
          r'Con.', '', r'PR', '', r'Cha.', r'Hdv', '', r'Bas.', r'GL',
          r'RD', r'Nat.', r'Vin.', '', '', r'Ch.v.']

    # cbar = fig.colorbar(cax, ticks=[vmin, -(h2 - h1 / 2), 0, np.max(DTT) / 2, np.max(DTT)])
    cbar = fig.colorbar(cax, ticks=[-360, -300, -240, -180, -120, -60,
                                    0, 60, 120, 180, 240, 300, 360])
    # cbar.ax.set_yticklabels([r'$\geq$ %.0f s' % -np.max(DTT), '0', '%.0f s' % np.max(DTT)])
    cbar.ax.set_yticklabels([r'$\geq$ %.0f' % 6, '5', '4', '3', '2', '1',
                             '0', '1', '2', '3', '4', '5', r'$\geq$ %.0f' % 6])
    cbar.ax.set_ylabel('Time in minutes', rotation=-90, va="bottom")
    # cbar.ax.set_yticklabels([int(vmin), int(vmin / 2), 0, int(np.max(DTT) / 2), int(np.max(DTT))])

    plt.xlabel('Destination', fontsize=22)
    plt.ylabel('Origin', fontsize=22)
    plt.yticks(range(25), SY)
    ax.xaxis.tick_top()
    # ax.xaxis.set_label_position('top')
    plt.xticks(range(25), SY, rotation=45, ha='left')
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.savefig('output/traveltime_{}_{}.pdf'.format(model,  time.strftime("%Y%m%d-%H%M%S")))
    plt.show()


if __name__ == '__main__':
    main()
