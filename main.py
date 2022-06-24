import random
import click
import os
import logging
import math
from pathlib import Path
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt
import numpy as np

SQRT_2 = math.sqrt(2)
SQRT_3 = math.sqrt(3)
SQRT_2PER3 = math.sqrt(2/3)
SHIFT = 2 * math.pi / 3

def create_logger(name='simulation'):
    """
    Create a logger
    :param name: log file name
    """
    fh = RotatingFileHandler(filename=f'{name}.log',
                             mode='w',
                             maxBytes=1024 * 1024,
                             backupCount=5,
                             encoding='utf-8')
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter(u'%(asctime)s %(filename)s:%(lineno)d %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def get_logger(name='simulation'):
    return logging.getLogger(name)


class SrfPllPiController:
    def __init__(self, cycles=10, noisy=False, imbalance=False):
        self.noisy = noisy
        self.imbalance = imbalance

        self.mf = CONFIG["mains_frequency"]
        self.sf = CONFIG["switch_frequency"] * 1000
        self.Ts = 1/self.sf
        self.CONST_S = 2*self.sf
        self.CONST_S2 = self.CONST_S * self.CONST_S

        self.cycles = cycles
        self.total_steps = cycles * (self.sf//self.mf)

        self.initial_phase = math.pi/3
        self.abc = np.zeros((self.total_steps, 3), dtype=float)
        self.dqz = np.zeros((self.total_steps, 3), dtype=float)
        self.theta = np.zeros((self.total_steps, 1), dtype=float)
        self.phi = np.zeros((self.total_steps, 1), dtype=float)

        self.phi_a = 0.0
        self.phi_b = -SHIFT
        self.phi_c = SHIFT

        self._Kp = 50.0
        self._Ki = 500.0
        self.step = 0

        '''
        The reason Kw is introduced because the calculate self.omega
        is just a approximation which is proportional to the genuine omega
        '''
        self._Kw = 1000.0
        self.omega_max = 2*math.pi*self.mf
        self.omega = np.full((self.total_steps, 1), 2*math.pi*50, dtype=float)
        self.delta_omega = np.zeros((self.total_steps, 1), dtype=float)
        self.theta_ref = np.zeros((self.total_steps, 1), dtype=float)

        '''
        SOGI
        '''
        self.k_sogi = 0.5
        self.dc, self.qc = self.calculate_DC_QC()

        self.abz = np.zeros((self.total_steps, 3), dtype=float)

        self.alpha = np.zeros((self.total_steps, 1), dtype=float)
        self.beta = np.zeros((self.total_steps, 1), dtype=float)
        self.zero = np.zeros((self.total_steps, 1), dtype=float)

        self.alpha_f_tmp = np.zeros((self.total_steps, 1), dtype=float)
        self.alpha_f = np.zeros((self.total_steps, 1), dtype=float)
        self.alpha_f_shifted = np.zeros((self.total_steps, 1), dtype=float)

        self.beta_f_tmp = np.zeros((self.total_steps, 1), dtype=float)
        self.beta_f = np.zeros((self.total_steps, 1), dtype=float)
        self.beta_f_shifted = np.zeros((self.total_steps, 1), dtype=float)

        self.zero_f_tmp = np.zeros((self.total_steps, 1), dtype=float)
        self.zero_f = np.zeros((self.total_steps, 1), dtype=float)
        self.zero_f_shifted = np.zeros((self.total_steps, 1), dtype=float)

        self.alpha_p = np.zeros((self.total_steps, 1), dtype=float)
        self.alpha_n = np.zeros((self.total_steps, 1), dtype=float)
        self.beta_p = np.zeros((self.total_steps, 1), dtype=float)
        self.beta_n = np.zeros((self.total_steps, 1), dtype=float)

        self.theta_ff = np.zeros((self.total_steps, 1), dtype=float)
        self.theta_diff = np.zeros((self.total_steps, 1), dtype=float)
        self.omega_ff = np.zeros((self.total_steps, 1), dtype=float)
        self.lpf_omega_c = 2*math.pi*10
        self.lpf_omega_a = self.lpf_omega_c*self.Ts/(1+self.lpf_omega_c*self.Ts)

        self.V_p = np.zeros((self.total_steps, 1), dtype=float)
        self.V_n = np.zeros((self.total_steps, 1), dtype=float)
        self.V_0 = np.zeros((self.total_steps, 1), dtype=float)

        self.theta_p = np.zeros((self.total_steps, 1), dtype=float)
        self.theta_n = np.zeros((self.total_steps, 1), dtype=float)
        self.theta_0 = np.zeros((self.total_steps, 1), dtype=float)

    @property
    def Kp(self):
        return self._Kp

    @Kp.setter
    def Kp(self, value=2.0):
        self._Kp = value

    @property
    def Ki(self):
        return self._Ki

    @Ki.setter
    def Ki(self, value=1.0):
        self._Ki = value

    @property
    def Kw(self):
        return self._Kw

    @Kw.setter
    def Kw(self, value=1.0):
        self._Kw = value

    def sim(self):

        for i in range(1, self.total_steps):
            self.sogi_track(i)

        xlim = self.cycles
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
        fig.subplots_adjust(hspace=0.5)

        ax1.set_xlabel('step')
        ax1.set_ylabel('ABC amplitude')
        ax1.set_xlim(0, xlim)
        ax1.grid(True)

        ax2.set_xlabel('step')
        ax2.set_ylabel('DQZ amplitude')
        ax2.set_xlim(0, xlim)
        ax2.grid(True)

        ax3.set_xlabel('step')
        ax3.set_ylabel('Angle')
        ax3.set_xlim(0, xlim)
        ax3.grid(True)

        ax4.set_xlabel('step')
        ax4.set_ylabel('ABZ')
        ax4.set_xlim(0, xlim)
        ax4.grid(True)

        t = np.arange(0.0, xlim, self.mf/self.sf)
        ax1.plot(t, self.abc, label=["a", "b", "c"])
        ax2.plot(t, self.dqz, label=["d", "q", "z"])
        ax3.plot(t, self.theta_ref, label="Theta ref")
        ax3.plot(t, self.theta_p, label="Theta P")
        ax3.plot(t, self.phi, label="Phi")
        ax3.plot(t, self.omega, label="Omega")
        #ax3.plot(t, self.omega_ff, label="Omega FF")
        #ax3.plot(t, self.alpha_p, label="Alpha P")
        #ax3.plot(t, self.beta_p, label="Beta P")
        ax4.plot(t, self.abz, label=["alpha", "beta", "zero"])

        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()
        plt.show()

    def calculate_voltage_abc(self, phase=0.0, sf=1000.0, step=0):
        """
        Generate three phase voltage with AGWN
        """
        if self.noisy:
            f = CONFIG["mains_frequency"] + random.uniform(-CONFIG["noise_f"], CONFIG["noise_f"])
            theta = 2 * math.pi * f * step * self.Ts
            theta_a = theta + phase
            theta_b = theta_a - SHIFT
            theta_c = theta_a + SHIFT

            voltage_a = math.sin(theta_a) + random.uniform(-CONFIG["noise_a"], CONFIG["noise_a"])
            voltage_b = math.sin(theta_b) + random.uniform(-CONFIG["noise_a"], CONFIG["noise_a"])
            voltage_c = math.sin(theta_c) + random.uniform(-CONFIG["noise_a"], CONFIG["noise_a"])
        else:
            f = CONFIG["mains_frequency"]
            theta = 2 * math.pi * f * step * self.Ts
            theta_a = theta + phase
            theta_b = theta_a - SHIFT
            theta_c = theta_a + SHIFT

            voltage_a = math.sin(theta_a)
            voltage_b = math.sin(theta_b)
            voltage_c = math.sin(theta_c)

        '''Test the case of lost balance'''
        if step > 1000 and self.imbalance:
            voltage_c = 0.7*voltage_c

        self.theta_ref[step] = theta_a % (2*math.pi)
        return voltage_a, voltage_b, voltage_c

    def calculate_abz(self, a=0.0, b=0.0, c=0.0):
        """
        Alpha Beta transform of a, b, c
        """
        alpha = (a - b / 2 - c / 2) * SQRT_2PER3
        beta = (b - c) / SQRT_2
        zero = (a + b + c) / SQRT_3

        return alpha, beta, zero

    def calculate_dqz(self, a=0.0, b=0.0, c=0.0, theta=0.0):
        """
        DQ0 transform of a, b, c and normalize
        """
        d = (a * math.sin(theta) + b * math.sin(theta - SHIFT) + c * math.sin(theta + SHIFT)) * (2/3)
        q = (a * math.cos(theta) + b * math.cos(theta - SHIFT) + c * math.cos(theta + SHIFT)) * (2/3)
        z = (a + b + c) / SQRT_3

        ''' Normalize '''
        r2 = math.sqrt(d*d + q*q)
        d = d/r2
        q = q/r2

        return d, q, z

    def dq_from_ab(self, alpha=0.0, beta=0.0, theta=0.0):
        d = math.cos(theta)*alpha + math.sin(theta)*beta
        q = -math.sin(theta)*alpha + math.cos(theta)*beta
        return d, q

    def extract_ps(self, a=0.0, b=0.0, c=0.0):
        """
        Extract the positive sequence from the grid
        :return: the positive sequence
        """

        a_p = a*math.cos(self.phi_a) + b*math.cos(self.phi_b + SHIFT) + c*math.cos(self.phi_c - SHIFT)
        b_p = a*math.sin(self.phi_a) + b*math.sin(self.phi_b + SHIFT) + c*math.sin(self.phi_c - SHIFT)

        v_p = math.sqrt(a_p*a_p + b_p*b_p)/3
        phi_p = math.atan(b_p/a_p)
        return v_p, phi_p

    def extract_ns(self, a=0.0, b=0.0, c=0.0):
        """
        Extract the negative sequence from the grid
        :return: the negative sequence
        """

        a_n = a*math.cos(self.phi_a) + b*math.cos(self.phi_b - SHIFT) + c*math.cos(self.phi_c + SHIFT)
        b_n = a*math.sin(self.phi_a) + b*math.sin(self.phi_b - SHIFT) + c*math.sin(self.phi_c + SHIFT)

        v_n = math.sqrt(a_n*a_n + b_n*b_n)
        phi_n = math.atan(b_n/a_n)
        return v_n, phi_n

    def extract_zs(self, a=0.0, b=0.0, c=0.0):
        """
        Extract the zero sequence from the grid
        :return: the zero sequence
        """

        a_z = a*math.cos(self.phi_a) + b*math.cos(self.phi_b) + c*math.cos(self.phi_c)
        b_z = a*math.sin(self.phi_a) + b*math.sin(self.phi_b) + c*math.sin(self.phi_c)

        v_z = math.sqrt(a_z*a_z + b_z*b_z)
        phi_z = math.atan(b_z/a_z)
        return v_z, phi_z

    def calculate_DC_QC_2nd(self, omega=1.0*2.0*math.pi*50):
        """
        https://lpsa.swarthmore.edu/LaplaceZTable/LaplaceZFuncTable.html
        :param omega:
        :return:
        """
        xi = 0.5*self.k_sogi
        xi_c = math.sqrt(1-xi*xi)
        omegaT = omega*self.Ts
        phi = math.acos(xi)
        EXP_TMP = math.exp(-xi*omegaT)

        DCa0 = 1.0
        DCa1 = -2*EXP_TMP*math.cos(xi_c*omegaT)
        DCa2 = EXP_TMP * EXP_TMP

        DCb0 = xi_c
        DCb1 = -2*EXP_TMP*math.sin(xi_c*omegaT + phi)
        DCb2 = 0.0

        dc = (DCa0, DCa1, DCa2, DCb0, DCb1, DCb2)
        pass

    def calculate_DC_QC(self, omega=1.0*2.0*math.pi*50):
        """
        DCa0, DCa1, DCa2, DCb0, DCb1, DCb2
        QCa0, QCa1, QCa2, QCb0, QCb1, QCb2
        :return:
        """
        omega2 = omega * omega
        kst = self.k_sogi*self.CONST_S*omega

        dividend_r = 1/(self.CONST_S2 + kst + omega2)

        DCa0 = 1.0
        DCa1 = (-2*self.CONST_S2 + 2*omega2)*dividend_r
        DCa2 = (self.CONST_S2 - kst + omega2)*dividend_r

        DCb0 = kst*dividend_r
        DCb1 = 0.0
        DCb2 = -DCb0

        dc = (DCa0, DCa1, DCa2, DCb0, DCb1, DCb2)

        QCa0 = 1.0
        QCa1 = DCa1
        QCa2 = DCa2

        QCb0 = self.k_sogi*omega2*dividend_r
        QCb1 = 2*QCb0
        QCb2 = QCb0

        qc = (QCa0, QCa1, QCa2, QCb0, QCb1, QCb2)

        return dc, qc

    def sogi_ds_qs(self, step):
        """
        Second order generalized integrator core
        :param step:
        :return:
        """
        '''Calculate alpha_f and alpha_f_shifted'''
        abz_a0 = self.abz[step][0]
        abz_a1 = self.abz[step-1][0] if step-1 > 0 else 0.0
        abz_a2 = self.abz[step-2][0] if step-2 > 0 else 0.0
        alpha_f_tmp_1 = self.alpha_f_tmp[step-1] if step-1 > 0 else 0.0
        alpha_f_tmp_2 = self.alpha_f_tmp[step-2] if step-2 > 0 else 0.0

        self.alpha_f_tmp[step] = self.dc[3]*abz_a0 + self.dc[4]*abz_a1 + self.dc[5]*abz_a2 - \
                                 self.dc[1]*alpha_f_tmp_1 - self.dc[2]*alpha_f_tmp_2

        alpha_f_tmp_0 = self.alpha_f_tmp[step]
        alpha_f_tmp_1 = self.alpha_f_tmp[step-1] if step-1 > 0 else 0.0
        alpha_f_tmp_2 = self.alpha_f_tmp[step-2] if step-2 > 0 else 0.0
        alpha_f_1 = self.alpha_f[step-1] if step-1 > 0 else 0.0
        alpha_f_2 = self.alpha_f[step-2] if step-2 > 0 else 0.0

        self.alpha_f[step] = self.dc[3]*alpha_f_tmp_0 + self.dc[4]*alpha_f_tmp_1 + self.dc[5]*alpha_f_tmp_2 - \
                             self.dc[1]*alpha_f_1 - self.dc[2]*alpha_f_2

        alpha_f_shifted_1 = self.alpha_f_shifted[step-1] if step-1 > 0 else 0.0
        alpha_f_shifted_2 = self.alpha_f_shifted[step-2] if step-2 > 0 else 0.0
        self.alpha_f_shifted[step] = self.qc[3]*alpha_f_tmp_0 + self.qc[4]*alpha_f_tmp_1 + self.qc[5]*alpha_f_tmp_2 - \
                                     self.qc[1]*alpha_f_shifted_1 - self.qc[2]*alpha_f_shifted_2

        '''Calculate beta_f and beta_f_shifted'''
        abz_b0 = self.abz[step][1]
        abz_b1 = self.abz[step-1][1] if step-1 > 0 else 0.0
        abz_b2 = self.abz[step-2][1] if step-2 > 0 else 0.0
        beta_f_tmp_1 = self.beta_f_tmp[step-1] if step-1 > 0 else 0.0
        beta_f_tmp_2 = self.beta_f_tmp[step-2] if step-2 > 0 else 0.0

        self.beta_f_tmp[step] = self.dc[3]*abz_b0 + self.dc[4]*abz_b1 + self.dc[5]*abz_b2 - \
                                self.dc[1]*beta_f_tmp_1 - self.dc[2]*beta_f_tmp_2

        beta_f_tmp_0 = self.beta_f_tmp[step]
        beta_f_tmp_1 = self.beta_f_tmp[step-1] if step-1 > 0 else 0.0
        beta_f_tmp_2 = self.beta_f_tmp[step-2] if step-2 > 0 else 0.0
        beta_f_1 = self.beta_f[step-1] if step-1 > 0 else 0.0
        beta_f_2 = self.beta_f[step-2] if step-2 > 0 else 0.0

        self.beta_f[step] = self.dc[3]*beta_f_tmp_0 + self.dc[4]*beta_f_tmp_1 + self.dc[5]*beta_f_tmp_2 - \
                            self.dc[1]*beta_f_1 - self.dc[2]*beta_f_2

        beta_f_shifted_1 = self.beta_f_shifted[step-1] if step-1 > 0 else 0.0
        beta_f_shifted_2 = self.beta_f_shifted[step-2] if step-2 > 0 else 0.0
        self.beta_f_shifted[step] = self.qc[3]*beta_f_tmp_0 + self.qc[4]*beta_f_tmp_1 + self.qc[5]*beta_f_tmp_2 - \
                                    self.qc[1]*beta_f_shifted_1 - self.qc[2]*beta_f_shifted_2

        '''Calculate zero_f and zero_f_shifted'''
        abz_z0 = self.abz[step][2]
        abz_z1 = self.abz[step-1][2] if step-1 > 0 else 0.0
        abz_z2 = self.abz[step-2][2] if step-2 > 0 else 0.0
        zero_f_tmp_1 = self.zero_f_tmp[step-1] if step-1 > 0 else 0.0
        zero_f_tmp_2 = self.zero_f_tmp[step-2] if step-2 > 0 else 0.0

        self.zero_f_tmp[step] = self.dc[3]*abz_z0 + self.dc[4]*abz_z1 + self.dc[5]*abz_z2 - \
                                self.dc[1]*zero_f_tmp_1 - self.dc[2]*zero_f_tmp_2

        zero_f_tmp_0 = self.zero_f_tmp[step]
        zero_f_tmp_1 = self.zero_f_tmp[step-1] if step-1 > 0 else 0.0
        zero_f_tmp_2 = self.zero_f_tmp[step-2] if step-2 > 0 else 0.0
        zero_f_1 = self.zero_f[step-1] if step-1 > 0 else 0.0
        zero_f_2 = self.zero_f[step-2] if step-2 > 0 else 0.0

        self.zero_f[step] = self.dc[3]*zero_f_tmp_0 + self.dc[4]*zero_f_tmp_1 + self.dc[5]*zero_f_tmp_2 - \
                            self.dc[1]*zero_f_1 - self.dc[2]*zero_f_2

        zero_f_shifted_1 = self.zero_f_shifted[step-1] if step-1 > 0 else 0.0
        zero_f_shifted_2 = self.zero_f_shifted[step-2] if step-2 > 0 else 0.0
        self.zero_f_shifted[step] = self.qc[3]*zero_f_tmp_0 + self.qc[4]*zero_f_tmp_1 + self.qc[5]*zero_f_tmp_2 - \
                                    self.qc[1]*zero_f_shifted_1 - self.qc[2]*zero_f_shifted_2

    def sogi_pnsc(self, step):
        self.alpha_p[step] = (self.alpha_f[step] - self.beta_f_shifted[step])*0.5
        self.alpha_n[step] = (self.alpha_f[step] + self.beta_f_shifted[step])*0.5

        self.beta_p[step] = (self.beta_f[step] + self.alpha_f_shifted[step])*0.5
        self.beta_n[step] = (self.beta_f[step] - self.alpha_f_shifted[step])*0.5

    def calculate_omega_ff(self, step=1):
        minimum = 1e-20
        if 0 > self.alpha_p[step] > -minimum:
            self.alpha_p[step] = -minimum
        elif minimum > self.alpha_p[step] >= 0.0:
            self.alpha_p[step] = minimum

        self.theta_ff[step] = math.atan(self.beta_p[step]/self.alpha_p[step])
        diff = self.theta_ff[step] - self.theta_ff[step-1]
        '''
        if diff < -math.pi:
            diff += 2*math.pi
        elif diff > math.pi:
            diff -= 2*math.pi
        else:
            pass
        '''

        self.theta_diff[step] = self.lpf_omega_a * diff + (1-self.lpf_omega_a)*self.theta_diff[step-1]
        self.omega_ff[step] = self.sf * self.theta_diff[step]

    def track(self, step=1):
        step_1 = step - 1

        self.abc[step][0], self.abc[step][1], self.abc[step][2] = \
            self.calculate_voltage_abc(self.initial_phase, self.sf, step)
        self.dqz[step][0], self.dqz[step][1], self.dqz[step][2] = \
            self.calculate_dqz(self.abc[step][0], self.abc[step][1], self.abc[step][2], self.theta[step_1])

        # u[n] = u[n - 1] + Kp * (e[n] - e[n - 1]) + Ki * Ts * e[n]
        self.phi[step] = self.dqz[step][1]
        self.omega[step] = self.omega[step_1] + \
            self.Kp * (self.phi[step] - self.phi[step_1]) + \
            self.Ki * self.Ts * self.phi[step]

        if self.omega[step] > self.omega_max:
            self.omega[step] = self.omega_max
        elif self.omega[step] < -self.omega_max:
            self.omega[step] = -self.omega_max
        else:
            # just keep it
            pass

        self.theta[step] = self.theta[step_1] + self.Ts * self.omega[step] * self.Kw
        self.theta[step] %= (2 * math.pi)

    def sogi_track(self, step=1):
        step_1 = step - 1

        '''Generate 3 phase voltage'''
        self.abc[step][0], self.abc[step][1], self.abc[step][2] = \
            self.calculate_voltage_abc(self.initial_phase, self.sf, step)

        '''aplha beta transform'''
        self.abz[step][0], self.abz[step][1], self.abz[step][2] = \
            self.calculate_abz(self.abc[step][0], self.abc[step][1], self.abc[step][2])

        '''SOGI'''
        # self.calculate_DC_QC_2nd(self.omega.item(step-1))
        self.dc, self.qc = self.calculate_DC_QC(self.omega.item(step-1))
        self.sogi_ds_qs(step)
        self.sogi_pnsc(step)

        self.calculate_omega_ff(step)

        dq = self.dq_from_ab(self.alpha_p[step], self.beta_p[step], self.theta_p[step_1])
        self.dqz[step][0] = dq[0]
        self.dqz[step][1] = dq[1]
        self.dqz[step][2] = self.abz[step][2]

        # u[n] = u[n - 1] + Kp * (e[n] - e[n - 1]) + Ki * Ts * e[n]
        self.phi[step] = self.dqz[step][1]
        pid_p = self.Kp * (self.phi[step] - self.phi[step_1])
        pid_i = self.Ki * self.Ts * self.phi[step]
        self.delta_omega[step] = self.delta_omega[step_1] + pid_p + pid_i

        self.omega[step] = self.delta_omega[step] + self.omega_ff[step]

        '''
        if self.omega[step] > self.omega_max:
            self.omega[step] = self.omega_max
        elif self.omega[step] < -self.omega_max:
            self.omega[step] = -self.omega_max
        else:
            # just keep it
            pass
        '''

        self.theta_p[step] = self.theta_p[step_1] + self.Ts * self.omega[step]
        self.theta_p[step] %= (2 * math.pi)

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(version='1.0.0')
def simulation():
    pass


@simulation.command()
@click.option('-s', '--switch_frequency', default=100, help='The switching frequency(KHz) of SiC or IGBT')
@click.option('-f', '--mains_frequency', default=50, help='The frequency(KHz) of AC(50 or 60Hz)')
@click.option('-n', '--cycles', default=10, help='The number of simulation cycles (20ms per cycle if 50Hz)')
def run(**kwargs):
    CONFIG["mains_frequency"] = kwargs['mains_frequency']
    CONFIG["switch_frequency"] = kwargs['switch_frequency']

    pll = SrfPllPiController(cycles=3, noisy=True, imbalance=True)
    pll.sim()


if __name__ == "__main__":
    launch_folder = Path(os.getcwd())
    script_folder = Path(os.path.dirname(os.path.realpath(__file__)))

    mylog = create_logger()
    mylog.info('The simulation is launched')

    CONFIG = {
        "project": 'The Power simulation',
        "version": 1.0,
        "noise_f": 0.001,
        "noise_a": 0.01,
    }

    simulation()

    mylog.info('The simulation ends')
