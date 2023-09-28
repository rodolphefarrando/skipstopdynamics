import time
import matplotlib.pyplot as plt
from .data import data
from .simulator import Simulator


def main():

    # choose wich model you want to simulate:
    # - first for the one where trains stop at all stations;
    # - second for the second model relaxing the OD constraint
    model = 'first' # or second
    name = 'unrestricted'

    # get the data from the file 'data/line1.csv'
    t1, t2, s1, s2, r1, r2, dist_stat, stations, skip_stations = data()
    n = len(t1)

    # Variable initialization
    headway1, headway2 = [], []
    frequency1, frequency2 = [0], [0]

    # choose the number of moving trains that you want for the simulations (last value is not included)
    start, end, step = 1, n, 1
    m = range(start, end, step)
    for m_ in m:
        # simulation1 is the simulation with no skip stop policy
        simulation1 = Simulator(m=m_, n=n, s=s1, t=t1, r=r1, model=model, horizon=min(m_*20, 600), dist_stat=dist_stat,
                                I=skip_stations, style='linear')
        h1, f1 = simulation1.simulation()
        headway1.append(h1)
        frequency1.append(f1)

        # simulation2 is the simulation with the one of the two skip stop policies
        simulation2 = Simulator(m=m_, n=n, s=s2, t=t2, r=r2, model=model, horizon=min(m_*20, 600), dist_stat=dist_stat,
                                I=skip_stations, style='linear')
        h2, f2 = simulation2.simulation()
        headway2.append(h2)
        frequency2.append(f2)

        print('m : ', m_, ' f1 = ', f1, ', f2 = ', f2)
        print('Delta f = ', f2 - f1)

    plt.figure(figsize=(12, 7))
    plt.plot(range(0, end, step), frequency1, label='All-stop policy')
    plt.plot(range(0, end, step), frequency2, label='Skip-stop policy, {} model'.format(name))

    # If you want the frequency on skipped stations
    # plt.plot(m, np.array(frequency2) / 2, label='freq. on low-demand stations')
    plt.ylabel('Frequency [train/h]', fontsize=24)
    plt.xlabel('Number of moving train', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim([0, 37])
    plt.xlim([0, n])
    plt.legend(fontsize=18)
    plt.savefig('output/simul_{}_{}.pdf'.format(model, time.strftime("%Y%m%d-%H%M%S")),
                bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
