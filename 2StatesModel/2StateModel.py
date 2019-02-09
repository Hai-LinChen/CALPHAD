import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from math import log, exp
from scipy.misc import derivative
from scipy.integrate import quad

"""
This program aims to demonstrate the 2-state model and it is based on 
the paper by Q. Chen and B. Sundman, J. Phase Equilibria 22 (6) (2001) 631-644.
"""

R = 8.314

class Magnetism:
    def __init__(self, TC, beta, p):
        self._TC = TC
        self._beta = beta
        self._p = p
        self._D = 0.33471979 + 0.49649686 * (1/p -1)
        self._RLN = R  * log(beta + 1)

    def ftao(self, T):
        TC = float(self._TC)
        t = T/TC
        p = float(self._p)
        if t <=1:
            f = 1 - (0.38438376 /t / p + 0.63570895 * (1/p -1 ) * (t**3/6 + t**9/135 + t**15/600 + t**21/1617)) / self._D
        else:
            f = - (t**(-7)/21 + t**(-21)/630 + t**(-35)/2975 + t**(-49)/8232) / self._D
        return f

    def CpMgn(self, T):
        TC = float(self._TC)
        t = T/TC
        p = float(self._p)
        if t <=1:
            g = (0.63570895 * (1/p -1) * (2*t**3 + 2/3 *t**9 + 2/5 *t**15+ 2/7 *t**21)) / self._D
        else:
            g = (2*t**(-7) + 2/3*t**(-21) + 2/5*t**(-35) + 2/7*t**(-49)) / self._D
        return self._RLN * g

    def GMO(self, T):           # Ordering energy, i.e. GMO(T) = Gdis(T) - Gdis(infinite)
        return self._RLN * T * self.ftao(T)

    def GMDO_infinite(self, T): # Disordering energy at infinite temperature, i.e. Gdis(infinite)
        return - self._RLN * (T - 0.38438376 * self._TC / self._p / self._D)

    def GMDO(self, T):          # Disordering energy at T, i.e. Gdis(T) = GMO(T) + Gdis(infinite)
        return self.GMO(T) + self.GMDO_infinite(T)

    def get_GMgnFunction(self):
        return self.GMDO

class TwoStates:
    def __init__(self, A, B, C, a, b, theta):
        self._A = A       # coefficients for evaluating G(liq) - G(amor)
        self._B = B
        self._C = C
        self._a = a       # coefficients for evaluating G(amor), using Einstein model
        self._b = b
        self._einstein = theta
        self.GMgnf = None # function for the magnetic contribution to the Gibbs energy

    def G_Einstein(self, T):    # contribution of Einstein function to Gibbs energy of full/ideal amorphous state
        Te = self._einstein
        return +1.5 * R * Te + 3 * R * T * log(1 - exp(- Te/T))

    def G_Magnetism(self, T):
        f = self.GMgnf          # by default, it is None, which means paramagnetism
        if f == None:
            return 0
        else:
            return f(T)

    def set_GMgnf(self, f):     # set the function of the magnetic contribution to Gibbs energy
        self.GMgnf = f

    def G_Amorphous(self, T):   # total Gibbs energy of full/ideal amorphous state
        return self._a + self._b * T**2 + self.G_Einstein(T) + self.G_Magnetism(T)

    def deltaG_ideal(self, T):  # Gibbs energy difference between ideal liquid and ideal amorphous state
        A = self._A
        B = self._B
        C = self._C
        return A + B*T + C*T*log(T)

    def deltaG_real(self, T):       # Gibbs energy difference between real state and ideal amorphous state
        return - R * T * log(1+ exp(-self.deltaG_ideal(T) / (R *T)))

    def G2SL(self, T):
        return self.G_Amorphous(T) + self.deltaG_real(T)

    def get_1st_derivative(self, T):
        return derivative(self.G2SL, T, dx=0.01)

    def get_entropy(self, T):
        S = -derivative(self.G2SL, T, dx=0.01)
        return S

    def get_enthalpy(self, T):
        return self.G2SL(T) + self.get_entropy(T) * T

    def get_heatcapacity(self, T):
        return derivative(self.get_enthalpy, T, dx = 0.01)

def CalculateGibbs(filename, qt, Ts):
    """
    Calculate Gibbs energy, entropy, enthalpy and heat capacity
    :param filename: name of the file to which results will be saved
    :param qt:       quantity with input parameters
    :param Ts:       temperature range in which calculations are to be made
    :return: G, S, H, Cp
    """
    lst_G, lst_S, lst_H, lst_Cp = [], [], [], []
    with open(filename, 'w') as f:
        f.write('$G   S   H   Cp\n')
        for T in Ts:
            G = qt.G2SL(T)
            lst_G.append(G)
            S = qt.get_entropy(T)
            lst_S.append(S)
            H = qt.get_enthalpy(T)
            lst_H.append(H)
            Cp = qt.get_heatcapacity(T)
            lst_Cp.append(Cp)
            f.write(str(G) + ' ' + str(S) + ' ' + str(H) + ' ' + str(Cp) + '\n')
    return lst_G, lst_S, lst_H, lst_Cp

def main():
    # parameters for magnetic contribution
    TC = 200
    beta = 1.70
    p = 0.25
    mgn = Magnetism(TC, beta, p)

    # parameters for energy difference between ideal amorphous and ideal liquid
    A = 4.27549478e+4
    B = -7.62400000
    C = -1.08230446

    # parameters for energy of the ideal amorphous state
    a = +7.1032080e+3
    b = -1.9730116e-3

    # filename, which the calculated results will be saved to and retrived from
    filename = "2StateModel_results.dat"
    Fe = TwoStates(A, B, C, a, b, theta = 245)
    Fe.set_GMgnf(mgn.get_GMgnFunction())
    Ts = np.arange(1, 5000)

    calculation = False     # or True, this is a switch, either to make a calculation or to retrive results from a file
    if calculation:
        lst_G, lst_S, lst_H, lst_Cp = CalculateGibbs(filename, Fe, Ts) # make the calculation
    else:
        lst_G, lst_S, lst_H, lst_Cp = np.loadtxt(filename, usecols=(0, 1, 2, 3), comments='$', delimiter=' ', unpack=True)

    # heat capacity in 2001Chen for comparison
    T_comp, Cp_comp = np.loadtxt('Cp_Fe_TC_2001Chen.txt', usecols=(0, 1), comments='$', delimiter=' ', unpack=True)

    fig = plt.figure(constrained_layout = True)
    gs = GridSpec(ncols=2, nrows=2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax1.plot(Ts, lst_G, 'r-', label ='liquid-amorphous')
    ax1.set_xlabel('Temperature, K')
    ax1.set_ylabel('Gibbs energy')
    ax1.legend(loc = 'best')
    ax2.plot(Ts, lst_S, 'b-', label ='liquid-amorphous')
    ax2.set_xlabel('Temperature, K')
    ax2.set_ylabel('Entropy')
    ax2.legend(loc = 'best')
    ax3.plot(Ts, lst_H, 'b-', label ='liquid-amorphous')
    ax3.set_xlabel('Temperature, K')
    ax3.set_ylabel('Enthalpy')
    ax3.legend(loc = 'best')
    ax4.plot(Ts, lst_Cp, 'b-', label ='liquid-amorphous')
    ax4.plot(T_comp, Cp_comp, 'ro', label ='as-published [2001Che]')
    ax4.set_xlabel('Temperature, K')
    ax4.set_ylabel('Heat capacity')
    ax4.legend(loc = 'best')
    plt.show()

    # understanding of the magnetic contribution
    # magnetic contribution can be regarded as either ordering or disordering, depending on
    # which reference state is used.
    # G_total = Gferromagn(T) + GMDO(T)
    # or
    # G_total = Gparamagn(T) + GMO(T)
    # The conversion of reference state can be made via
    # Gparamagn = Gferromagn + GMDO(T=inf)
    # then,
    # GMO(T) = GMDO(T) - GMDO(inf)
    #
    # The model adopted in this paper is to provide the possibility of converting the reference state.
    # The parameters are actually given, with the reference state of ferromagn, although
    # GMO and GMDO(inf) are also provided. This is because GMO + GMDO(inf) summes up to GMDO(T).
    #
    # Anyway, the presence of the two terms allows us to check the contribution of each.
    #
    # Gferromagn can be obtained via experiments
    # Gparamagn may be obtained via theoretical calculation
    # Their difference gives the GMDO(T=inf)
    #
    # Practically, the The difference is evaluated via the magnetic heat capacity.
    #          ds
    # Cp = T -----
    #         dT
    #          |
    #         | |                                             _ -- -- -- -- -- --
    #        |   |                                          +
    #       |     |                                       /
    #      |       |                                    /
    #     |         |                                 /         S_Mgn
    #    |    Cp     |                               /
    #   |    Mgn      |                             /
    #  |               + _ _ _ _ _ _ _ _ _ _       /
    #
    #  integral of Cp_mgn/T gives S_mgn, which becomes a constant while T is sufficiently large
    #  integral of S_mgn give G_mgn, which linearly depends on temperature when T is sufficiently large
    #
    #  SMgn(infinite) is an integration of Cp_Mgn in the interval of (0, oo)
    #  GMDO(infinite) is actually an integration of S_Mgn(infinite) in the interval of (0, T)
    #
    #  This might be very much confusing. One can have a better understanding with the aid of the following figure.
    #  Naturally, one can use Gferromgn as reference, then GMDO, which can be calculated from Cp_Mgn, is added to it.
    #  At low temperatures, GMDO is close to 0, as magnetic state is stable. Around TC, GMDO starts to be considerable.
    #  At sufficiently high temperatures, GMDO becomes a straight line. This can be accepted without problem.
    #
    #  Now, one may want to use Gparamgn as reference and then evaluate GMO (order), instead of GMDO (disorder). How?
    #  Well, there is a straight line of GMDO. Use it, i.e. GMDO(infinite), as the reference. And extrapolate it down
    #  down to RT (SER) or 0 K. Note that GMDO(inf) is an integation over (0, T) of an integration over (0, oo)
    #  of magnetic heat capacity, with a minus sign. One may say GMDO(infinite) is kind of arbitrary.
    #  After such a conversion, however, GMD seems to be very meaningful, since it reduces from a maximum value at 0 K
    #  to zero at temperatures above TC, where the matarial is paramagnetic. Probably there is a typo or mistake in the
    #  publications.

    Ts = np.arange(1, 5000)
    lst_GMO, lst_GMDO, lst_GMDO_inf, lst_CpMgn = [], [], [], []
    for Ti in Ts:
        # GMO = GMDO - GMDO_inf
        # GMDO_inf and GMDO are related to each other
        GMO_i = mgn.GMO(Ti)   # ordering energy
        GMDO_i = mgn.GMDO(Ti) # disordering energy at T
        GMDO_inf_i = mgn.GMDO_infinite(Ti)    # disordering energy at infinite temperature
        GCpMgn = mgn.CpMgn(Ti)
        lst_GMO.append(GMO_i)
        lst_GMDO.append(GMDO_i)
        lst_GMDO_inf.append(GMDO_inf_i)
        lst_CpMgn.append(GCpMgn)
    plt.plot(Ts, lst_GMO, '+', label= 'Magnetic ordering')
    plt.plot(Ts, lst_GMDO, '.', label= 'Magnetic disordering')
    plt.plot(Ts, lst_GMDO_inf, '-', label= 'Magnetic disordering, infinite')
    plt.xlabel('Temperature, K')
    plt.ylabel('Magnetic contribution to Gibbs energy')
    plt.legend(loc = 'best')
    plt.show()

    plt.plot(Ts, lst_CpMgn, 'ro', label='Magnetic heat capacity')
    plt.plot(Ts, lst_Cp, 'b-', label='total heat capacity')
    plt.xlabel('Temperature, K')
    plt.ylabel('Magnetic contribution to Cp')
    plt.legend(loc = 'best')
    plt.show()

main()