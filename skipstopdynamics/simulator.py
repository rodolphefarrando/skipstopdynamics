import random
import numpy as np
from .trajectories import plottrajectories
from .initial_positions import initial_position

MINUS_INF = -1e8
LIM_H = 400


class Simulator(object):
    """Simulation of traffic
    """

    def __init__(self, m, n, s, t, r, w=None, model='first', horizon=200, dist_stat=None, plot=False, I=None,
                 stations=None, style='linear', tt=False):
        """

        :param m: number of trains
        :param n: number of segments
        :param s: safe separation time
        :param t: travel time
        :param r: = t + w
        :param w: waiting time
        :param model: choose your model
        :param horizon: Nb of iteration
        :param dist_stat: Distance between stations
        :param plot: Plot or not plot
        :param I: set of stations that can be skipped
        :param style: initial positions style
        :param stations: Stations with mandatory stop
        :param tt: if you want to have the information relative to travel time set to true (return D and A instead of f)
        """
        self.w = np.array(w) if w is not None else w
        self.r = np.array(r)
        self.t = np.array(t)
        self.s = np.array(s)
        self.m = m
        self.n = n
        self.I = I
        if model not in ['first', 'second']:
            print('The seleted model does not exist, please choose "first" or "second"; first assigned')
            model = 'first'
        self.model = model

        self.i0 = self.I[0]
        self.i1 = self.I[int(len(self.I) / 2)]

        if len(self.t.shape) > 1:
            self.cycle = self.t.shape[1]
        else:
            self.cycle = 1
            self.t = np.reshape(self.t, (self.t.shape[0], 1))
            self.s = np.reshape(self.s, (self.s.shape[0], 1))
            self.r = np.reshape(self.r, (self.r.shape[0], 1))

        self.b, self.bbar = initial_position(n=n, m=m, I=I, style=style, model=model)

        if horizon == 0:
            self.horizon = 200
        else:
            self.horizon = horizon

        self.plot = plot
        self.dist_stat = dist_stat
        self.stations = stations
        self.tt = tt

    def plottraj(self, D, A):
        """
        plot the trajectories of the trains
        """
        plottrajectories(D, A, self.b, self.m, self.dist_stat, self.model)

    def case(self, j, k):
        """
        return the case l for the first model
        """
        beg = self.i0 if j <= self.n / 2 else self.i1
        if j in self.I:
            if k % 2 == 0:
                l = 1 if sum(self.b[beg+1:j + 1]) % 2 == 0 else 0
            else:
                l = 0 if sum(self.b[beg+1:j + 1]) % 2 == 0 else 1
        elif j - 1 in self.I:
            if k % 2 == 0:
                l = 1 if sum(self.b[beg+1:j]) % 2 == 0 else 0
            else:
                l = 0 if sum(self.b[beg+1:j]) % 2 == 0 else 1
        else:
            l = 0

        return l

    def simulation(self):
        """
        Do the simulation
        return: the headway and frequency if nothing is specified; the matrix of departure and arrival if asked
        """

        # Initialization
        k = 1
        D = MINUS_INF * np.ones((self.n, self.horizon))
        A = MINUS_INF * np.ones((self.n, self.horizon))
        D[:, 0] *= self.bbar
        A[:, 0] *= self.bbar
        for j in range(self.n - 1):
            A[j, 0] = D[j, 0] + self.r[j, 0]

        # simulation on k; the number of interation
        while k < self.horizon:
            err = 1e10
            step = 0

            # Determine l ;
            # in the case of the first model l depends on j and is determined at the beginning of the j loop
            l = 0 if self.cycle == 1 else (k - 1) % 2

            # for each k we need to compute the departures multiple times until we reach the convergence
            while err > 1e-6 and step <= 500:
                # If there is no convergence print the error
                if step == 500:
                    print('err', err)

                t1 = np.zeros(self.n)
                t2 = np.zeros_like(t1)
                d_old = np.copy(D[:, k])

                # Loop on all the nodes of the line;
                # mandatory because for the first model we determine the value of l for each node independently
                for j in range(self.n):

                    # l for the first model; else already assigned at the beginning of the k loop
                    if self.model == 'first' and self.cycle != 1:
                        l = self.case(j, k)

                    # In python when you put a minus before the index, the counting will start at the end of the array;
                    # when j=0 D[j-1, k] = D[-1, k] will select the last row of kth column of D
                    # This is why there is no particular case for j=0;
                    t1[j] = D[j - 1, k - self.b[j - 1]] + self.t[j - 1, l]
                    if j != self.n - 1:
                        t2[j] = D[j + 1, k - self.bbar[j]] + self.s[j + 1, l]
                        A[j, k] = D[j, k] + self.r[j, l]
                    else:
                        t2[j] = D[0, k - self.bbar[j]] + self.s[0, l]
                        A[j, k] = D[j, k] + self.r[j, l]

                # Assign the new values to D^k
                D[:, k] = np.maximum(t1, t2)
                # Compute the difference between the old values and the new ones
                err = np.linalg.norm(D[:, k] - d_old)
                step += 1

            k += 1

        # Get the convergence values
        if LIM_H >= k - 1:
            h0 = (D[:, k - 1]) / (k - 1)
        else:
            h0 = (D[:, k - 1] - D[:, k - LIM_H]) / LIM_H

        # The headway is the mean of h
        h0 = h0.mean()

        # Frequency
        # f = 1 / h (*3600 to have it in trains per hour)
        f0 = 3600 / h0

        if self.plot:
            self.plottraj(D, A)

        # Will return the departure and arrival time matrices
        if self.tt:
            return D, A, h0

        return h0, f0
