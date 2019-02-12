import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.cm import get_cmap
from math import log, exp
from scipy.misc import derivative
from scipy.integrate import quad

"""
This program aims to 
(1) demonstrate the 2-state model, based on Q. Chen and B. Sundman, J. Phase Equilibria 22 (6) (2001) 631-644.
(2) illustrate the contribution of magnetism to Cp (heat capacity), S (entropy), H (enthalpy) & G (Gibbs energy)
"""

R = 8.314                   # gas constant, (m3 ⋅ Pa) / (K ⋅ mol)

class Magnetism:
    """
    >> Usually, magnetism occurs at (relatively) low temperatures, vanishes with increasing temperature
    and disappears at a critical temperature T*.

    *1* The magnetic heat capacity has a "lambda" shape. It attains its maximum at TC and becomes zero above T*.

    *2* The magnetic entropy can be integrated from Cp_Mgn / T. It increases with increasing temperature and
    reaches a constant at T* and above.

    *3* The magnetic enthalpy can be integrated from Cp_Mgn. It increases with increasing temperature and
    reaches a constant at T* and above.

    *4* The magnetic energy can be evaluated as the sum of H_Mgn and (- (T * S_Mgn)). It decreases with increasing
    temperature and becomes linearly dependent on temperature above T*.


    >> Energy is a relative quantity. Its value depends on the choice of reference state.

    * If one choose the ferromagnetic state at 25 C as reference, then magnetic energy will correspond to disordering.
    The magnetic disordering energy is what is defined in item *4* and can be evaluated from the following equation,

        G_Mgn_diso = H_Mgn and (- (T * S_Mgn))

    where S_Mgn and H_Mgn can be directly obtained via integrations, respectively, item *2* and item *3*.

    It has a value of Zero at 0 K and gradually decreases with increasing temperature. It becomes linearly dependent
    on temperature above T* (i.e. paramagnetic).

    That straight line can be regarded as the asymptote of the curve of G_Mgn_diso. It is also known as G_Mgn_diso(oo),
    which is quite confusing though. The definite of G_Mgn_diso(oo) in the publication is wrong. In fact, it can be
    obtained via
        G_Mgn_diso(oo) = H_Mgn(oo) - (T * S_Mgn(oo))

    where S_Mgn(oo) and H_Mgn(oo) can be directly obtained via integrations from 0 K to oo K, respectively, item *2*
    and item *3*.

    * If one choose the paramagnetic state as reference, then magnetic energy will correspond to ordering, G_Mgn_ord.
        G = Gferro + G_Mgn_diso
          = Gferro + G_Mgn_diso (oo) + G_Mgn_diso - G_Mgn_diso (oo)
          = Gpara + G_Mgn_ord

        obviously,
        Gpara = Gferro + G_Mgn_diso (oo)

        G_Mgn_ord = G_Mgn_diso - G_Mgn_diso (oo)
    which implies that G_Mgn_ord can be obtained from G_Mgn_diso by rotating and shifting its curve until its asymptote
    overlaps with the x axis (i.e. y = 0).

    The magnetic ordering energy has the most negative value at 0 K and it is the same as H_Mgn(oo) but with a '-' sign,
        G_Mgn_ord(0 K) = - H_Mgn(oo).

    >> Practically, one can analytically do the integration and derive explicit expressions of S_Mgn and H_Mgn, as well
    as G_Mgn, in order to avoid doing integrations numerically.

    * Results from the analytical expressions will be compared with those from numertical integrations, in order to
    validate the model and the program.
    """
    def __init__(self, TC, beta, p):
        self._TC = TC                       # Curie temperature
        self._beta = beta                   # Magnetic moment
        self._p = p
        self._D = 0.33471979 + 0.49649686 * (1/p -1)
        self._RLN = R  * log(beta + 1)
        self._SMgn_infinite = self.SMgn_integration(5000)   # 5000 is considered as sufficiently large
        self._HMgn_infinite = self.HMgn_integration(5000)   #

    def CpMgn(self, T):            # Cp (heat capacity) due to magnetism
        TC = self._TC
        t = T/TC
        p = self._p
        if t <=1:
            g = (0.63570895 * (1/p -1) * (2*t**3 + 2/3 *t**9 + 2/5 *t**15+ 2/7 *t**21)) / self._D
        else:
            g = (2*t**(-7) + 2/3*t**(-21) + 2/5*t**(-35) + 2/7*t**(-49)) / self._D
        return self._RLN * g

    def SMgn_integration(self, T):  # evaluation of S_Mgn, via integration
        def integral(T):
            return self.CpMgn(T)/T
        return quad(integral, 0.001, T)[0]

    def HMgn_integration(self, T):  # evaluation of H_Mgn, via integration
        def integral(T):
            return self.CpMgn(T)
        return quad(integral, 0.001, T)[0]

    def get_SMgn_inf(self, T):      # total magnetic contribution to entropy, S
        return self._SMgn_infinite

    def get_HMgn_inf(self, T):      # total magnetic contribution to enthalpy, H
        return self._HMgn_infinite

    def GMgn_dis_inf_integration(self, T):
        def integral(T):
            return - self._SMgn_infinite
        return self._HMgn_infinite + quad(integral, 0.001, T)[0]

    # def GMgn_integration(self, Ts):
    #     lst_SMgn = {}
    #     lst_GMgn = []
    #     for Ti in Ts:
    #         lst_SMgn[Ti] = self.SMgn_integration(Ti)
    #     def integral(T):
    #         return - lst_SMgn[T]               # negative entropy
    #     for Ti in Ts:
    #         lst_GMgn.append(quad(integral, 0.001, Ti)[0])  # This does not work!!!
    #     return lst_GMgn

    # Analytical expressions for magnetic energy terms
    def GMgn_ord(self, T):           # Magnetic ordering energy, analytical expression
        TC = float(self._TC)
        t = T / TC
        p = float(self._p)
        if t <= 1:
            f = 1 - (0.38438376 / t / p + 0.63570895 * (1 / p - 1) * (
            t ** 3 / 6 + t ** 9 / 135 + t ** 15 / 600 + t ** 21 / 1617)) / self._D
        else:
            f = - (t ** (-7) / 21 + t ** (-21) / 630 + t ** (-35) / 2975 + t ** (-49) / 8232) / self._D
        return self._RLN * T * f

    def GMgn_dis_infinite(self, T):  # Infinite mggnetic disordering energy, analytical expression
        return - self._RLN * (T - 0.38438376 * self._TC / self._p / self._D)

    def GMgn_dis(self, T):           # Magnetic disordering energy, analytical expression
        return self.GMgn_ord(T) + self.GMgn_dis_infinite(T)

    def get_GMgn_dis_Function(self):
        return self.GMgn_dis

class TwoStates:
    """
    In order to allow independent evaluation of magnetic contribution, a switch is used.
    """
    def __init__(self, A, B, C, a, b, theta, force_paramagn = False):
        self._A = A       # coefficients for evaluating G(liq) - G(amor)
        self._B = B
        self._C = C
        self._a = a       # coefficients for evaluating G(amor), using Einstein model
        self._b = b
        self._einstein = theta
        self.f_GMgn_dis = None # function for the magnetic contribution to the Gibbs energy
        self._force_paramagn = force_paramagn

    def G_Einstein(self, T):   # contribution of Einstein function to Gibbs energy of full/ideal amorphous state
        Te = self._einstein
        return +1.5 * R * Te + 3 * R * T * log(1 - exp(- Te/T))

    def G_Magn_dis(self, T):    # function for magnetic disordering energy
        f = self.f_GMgn_dis      # by default, it is None, which means paramagnetism
        if f == None:
            return 0
        else:
            return f(T)

    def set_Mgn(self, Mgn):
        self._Mgn = Mgn
        self.f_GMgn_dis = Mgn.get_GMgn_dis_Function()

    def set_force_paramagn(self, False_or_True):
        self._force_paramagn = False_or_True

    def G_Amorphous(self, T):    # total Gibbs energy of full/ideal amorphous state
        if self._force_paramagn == False:     # total Gibbs energy including magnetic contribution
            return self.G_Einstein(T) + self._a + self._b * T**2 + self.G_Magn_dis(T)
        elif self._force_paramagn == True:    # Gibbs energy for paramagnetic state
            return self.G_Einstein(T) + self._a + self._b * T**2 + self._Mgn.GMgn_dis_infinite(T)

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

def AnalyticalCalculation(filename, qt, Ts):
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
    tab10 = get_cmap('tab10')
    colors = [tab10(i/9) for i in range(10)]

    # parameters for magnetic contribution
    TC = 200                     # Curie temperature
    beta = 1.70                  # magnetic moment
    p = 0.25
    mgn = Magnetism(TC, beta, p) # define an instance

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
    Fe.set_Mgn(mgn)
    Ts = np.arange(1, 5000, 15)  # temperature range from 1 K to 5000 K

    # calculation using analytical equations
    calculation = True           # this is a switch, either to make a calculation or to retrive results from a file
    if calculation:
        lst_G, lst_S, lst_H, lst_Cp = AnalyticalCalculation(filename, Fe, Ts) # make the calculation
    else:
        lst_G, lst_S, lst_H, lst_Cp = np.loadtxt(filename, usecols=(0, 1, 2, 3), comments='$', delimiter=' ', unpack=True)

    # heat capacity in 2001Chen for comparison
    T_comp, Cp_comp = np.loadtxt('Cp_Fe_TC_2001Chen.txt', usecols=(0, 1), comments='$', delimiter=' ', unpack=True)
    T2_Comp, H_comp = np.loadtxt('H_Fe_TC_2001Chen.txt', usecols=(0, 1), comments='$', delimiter=' ', unpack=True)

    Fe.set_force_paramagn(False_or_True= True) # force to get rid of ferromagnetism
    lst_G_paramgn = []
    for Ti in Ts:
        lst_G_paramgn.append(Fe.G2SL(Ti))

    fig1 = plt.figure(constrained_layout = True)
    gs = GridSpec(ncols=2, nrows=2, figure=fig1)
    ax1 = fig1.add_subplot(gs[0, 0])
    ax2 = fig1.add_subplot(gs[0, 1])
    ax3 = fig1.add_subplot(gs[1, 0])
    ax4 = fig1.add_subplot(gs[1, 1])
    ax1.plot(Ts, lst_G, '-', color=colors[0], label ='liquid-amorphous')
    ax1.plot(Ts, lst_G_paramgn, '^', color=colors[5], label='paramagnetic state')
    ax1.set_xlim(0, 2000)
    ax1.set_ylim(-50000, +20000)
    ax1.set_xlabel('Temperature, K')
    ax1.set_ylabel('Gibbs energy')
    ax1.legend(loc = 'best')
    ax2.plot(Ts, lst_S, '-', color=colors[1], label ='liquid-amorphous')
    ax2.set_xlabel('Temperature, K')
    ax2.set_ylabel('Entropy')
    ax2.legend(loc = 'best')
    ax3.plot(Ts, lst_H, '-', color=colors[2], label ='liquid-amorphous')
    ax3.plot(T2_Comp, H_comp, '^', color=colors[5], label ='as-published [2001Che]')
#    ax3.set_xlim(1600, 2400)
#    ax3.set_ylim(40000, 100000)
    ax3.set_xlabel('Temperature, K')
    ax3.set_ylabel('Enthalpy')
    ax3.legend(loc = 'best')
    ax4.plot(Ts, lst_Cp, '-', color=colors[3], label ='liquid-amorphous')
    ax4.plot(T_comp, Cp_comp, 'o', color=colors[4], label ='as-published [2001Che]')
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
    Fe.set_force_paramagn(False_or_True=False)
    fig2 = plt.figure(constrained_layout=True)
    gs2 = GridSpec(nrows=1, ncols=2, figure = fig2)
    axn1 = fig2.add_subplot(gs2[0, 0])
    axn2 = fig2.add_subplot(gs2[0, 1])

    # Evaluation of magnetic energy components and Comparison between analytical equations and numerical calculations
    Ts_shift = np.arange(25, 5025, 50)
    lst_GMO, lst_GMDO, lst_GMDO_inf, lst_CpMgn = [], [], [], []
    lst_GMgn, lst_GMgn_inf = [], []
    for Ti in Ts:
        GMO_i = mgn.GMgn_ord(Ti)                  # magnetic ordering energy
        GMDO_i = mgn.GMgn_dis(Ti)                 # magnetic disordering energy
        GMDO_inf_i = mgn.GMgn_dis_infinite(Ti)    # infinite magnetic disordering energy
        GCpMgn = mgn.CpMgn(Ti)
        lst_GMO.append(GMO_i)
        lst_GMDO.append(GMDO_i)
        lst_GMDO_inf.append(GMDO_inf_i)
        lst_CpMgn.append(GCpMgn)

    for Ti in Ts_shift:
        GMgn_inf_i = mgn.GMgn_dis_inf_integration(Ti)  #
        lst_GMgn_inf.append(GMgn_inf_i)

    Fe.set_force_paramagn(False_or_True= True) # force to get rid of ferromagnetism
    lst_Cp_paramgn = []
    for Ti in Ts:
        lst_Cp_paramgn.append(Fe.get_heatcapacity(Ti))

    axn1.plot(Ts, lst_GMO, '+', color=colors[0], label= 'Magnetic ordering, analytical')
    axn1.plot(Ts, lst_GMDO, '.', color=colors[1], label= 'Magnetic disordering, analytical')
    axn1.plot(Ts, lst_GMDO_inf, '-', color=colors[2], label= 'Infinite magnetic disordering, analytical')
    # plt.plot(Ts_reduced, lst_GMgn, 'o', label = 'integrated GMgn')
    axn1.plot(Ts_shift, lst_GMgn_inf, '+', color=colors[3], label = 'Infinite magnetic disordering, integrated')
    axn1.set_xlabel('Temperature, K')
    axn1.set_ylabel('Magnetic contribution to Gibbs energy')
    axn1.legend(loc = 'best')
    axn1.set_xlim(0, 2000)
    axn1.set_ylim(-15000, 5000)

    axn2.plot(Ts, lst_CpMgn, 'o', color=colors[5], label='Magnetic heat capacity')
    axn2.plot(Ts, lst_Cp, '-', color=colors[6], label='total heat capacity')
    axn2.plot(Ts, lst_Cp_paramgn, '+', color=colors[7], label='heat capacity of paramagnetic state')
    axn2.set_xlabel('Temperature, K')
    axn2.set_ylabel('Magnetic contribution to Cp')
    axn2.legend(loc = 'best')
    plt.show()

main()