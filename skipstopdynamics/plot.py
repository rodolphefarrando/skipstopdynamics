from skipstopdynamics.data import data
from skipstopdynamics.simulator import Simulator


def main():

    model = 'second'  # or # second
    name = 'unrestricted' if model == 'second' else 'restricted'

    t1, t2, s1, s2, r1, r2, dist_stat, stations, skip_stations = data()
    # skip_stations = sorted(list(set(stations) ^ set(skip_stations)))
    n = len(t1)
    print('Number of segments on the line:', n, ', Model', name)

    m_opt = 40

    simulation1 = Simulator(m=m_opt, n=n, s=s1, t=t1, r=r1, model=model, horizon=200, dist_stat=dist_stat,
                            I=skip_stations, plot=False)

    # simulation2 is the simulation with the one of the two skip stop policies
    simulation2 = Simulator(m=m_opt, n=n, s=s2, t=t2, r=r2, model=model, horizon=200, dist_stat=dist_stat,
                            I=skip_stations, plot=True)

    print('SIM 1')
    h, f1 = simulation1.simulation()

    print('SIM 2')
    h, f2 = simulation2.simulation()

    # print(f2)
    print(f1, f2)


if __name__ == '__main__':
    main()
