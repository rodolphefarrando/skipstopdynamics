import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)


def plottrajectories(D, A, b, m, dist_station, model):

    # D_ = D[:, 1:m+1]
    # A_ = A[:, 1:m+1]

    D_ = D[:, -m:]
    A_ = A[:, -m:]

    err_a = 0

    n = len(b)
    cum_b = np.cumsum(b)

    # D -= dmin
    station = {}
    traj = {}
    begtraj = {}
    wnew = {}
    for j in range(1, m + 1):
        i = 0
        traj[j] = np.zeros((2 * len(b), 1))
        station[j] = np.zeros((2 * len(b), 1))
        while cum_b[i] != j:
            i += 1

        i *= 2
        traj[j][i] = D[int(i / 2), -m - 1]
        traj[j][i + 1] = A[int(i / 2), -m - 1]

        if dist_station is None:
            station[j][i] = i / 2
            station[j][i + 1] = i / 2 + 1
        else:
            station[j][i] = dist_station[int(i / 2)]
            station[j][i + 1] = dist_station[int(i / 2) + 1]
        begtraj[j] = i
        for k in range(1, n):
            kb = 2 * k
            if (kb + i) < (2 * n):
                col = kb + i
            else:
                col = kb + i - 2 * n

            traj[j][col] = D_[int(col / 2), cum_b[int(col / 2) - 1] - j]
            # traj[j][col + 1] = D_[int(col / 2), cum_b[int(col / 2) - 1] - j]
            if int(col / 2) == A_.shape[0] - 1:
                traj[j][col + 1] = A_[int(col / 2), cum_b[int(col / 2) - 1] - j]
            else:
                traj[j][col + 1] = A_[int(col / 2), cum_b[int(col / 2) - 1] - j]

            # if int(col / 2) - 1 in stations:
            #     traj[j][col - 1] = traj[j][col]

            if traj[j][col] - traj[j][col - 1] < 0:
                err_a += 1
                # traj[j][col - 1] = traj[j][col]

            if dist_station is None:
                station[j][col] = col / 2
                station[j][col + 1] = (col / 2 + 1)
            else:
                station[j][col] = dist_station[int(col / 2)]
                station[j][col + 1] = dist_station[int(col / 2 + 1)]

        wnew[j] = traj[j][1:] - traj[j][:-1]
    print('nb of errors :', err_a)
    plt.figure(figsize=(12, 7))
    color = ['b', 'k']
    col = 0
    lstyle = ['-', '--']
    for j in range(1, m + 1):
        if j == 1:
            label = 'First Service'
        elif j == 2:
            label = 'Second Service'
        else:
            label = ''
        # plt.plot(traj[j][begtraj[j]:], station[j][begtraj[j]:],
        #          traj[j][:begtraj[j]], station[j][:begtraj[j]], color=color[col])
        end = n if begtraj[j] > n else begtraj[j]
        plt.plot(traj[j][:end], station[j][:end], color=color[j % 2], linestyle=lstyle[j % 2],
                 label=label)
        plt.plot(traj[j][end:n], station[j][end:n], color=color[j % 2], linestyle=lstyle[j % 2],
                 label='')

        # plt.plot(traj[j][:begtraj[j]], range(0, begtraj[j]))
        col += 1
        if col >= len(color):
            col = 0


    yticksdist = np.array([0, 1000, 1800, 2800, 3700, 4400, 5099, 5750, 6400, 6900, 7550, 8300,
                           8800, 9200, 9800, 10340, 10990, 11690, 12790, 13990, 14840, 16339, 16990,
                           17640, 18590]) + 600

    plt.yticks(yticksdist,
               [r'\bf{Def.}', '\it{E. Def}', '\it{Neuilly}', '\it{Sablons}', r'\bf{Maillot}',
                '\it{Arg.}', r'\bf{CDG}', '\it{Geo. V}', r'\bf{Roos.}', r'\bf{CE}', r'\bf{Conc.}',
                '\it{Tuil.}', r'\bf{PR}', '\it{Louvre}', r'\bf{Chat.}', r'\bf{HdV}', '\it{SP}',
                r'\bf{Bast.}', r'\bf{GdL}', r'\bf{RD}', r'\bf{Nation}', r'\bf{PdV}', '\it{SM}',
                '\it{Ber}', r'\bf{CDV}'])

    # plt.xticks([26000 + 600 * i for i in range(7)], [0, 10, 20, 30, 40, 50, 60])
    # plt.xlim([26000, 26000 + 600*6])
    plt.xticks([25000 + 600 * i for i in range(7)], [0, 10, 20, 30, 40, 50, 60])
    plt.xlim([25000, 25000 + 600*6])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Time in minutes', fontsize=24)
    plt.ylabel('', fontsize=24)
    plt.legend(loc='upper left', fontsize=18)
    plt.ylim([0, 10340])
    plt.grid()
    # plt.savefig('output/traj_main.pdf')
    plt.savefig('output/traj_{}_{}.pdf'.format(model, time.strftime("%Y%m%d-%H%M%S"))
                , bbox_inches='tight')
    plt.show()
