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
        self.cycles = cycles
        self.total_steps = cycles * (self.sf//self.mf)

        self.initial_phase = math.pi/3
        self.abc = np.zeros((self.total_steps, 3), dtype=float)
        self.dqz = np.zeros((self.total_steps, 3), dtype=float)
        self.theta = np.zeros((self.total_steps, 1), dtype=float)
        self.phi = np.zeros((self.total_steps, 1), dtype=float)

        self._Kp = 50.0
        self._Ki = 500.0
        self.step = 0

        '''
        The reason Kw is introduced because the calculate self.omega
        is just a approximation which is proportional to the genuine omega
        '''
        self._Kw = 1000.0
        self.omega_max = 2*math.pi*self.mf
        self.omega = np.zeros((self.total_steps, 1), dtype=float)
        self.theta_ref = np.zeros((self.total_steps, 1), dtype=float)

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
            self.track(i)

        xlim = self.cycles
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
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

        t = np.arange(0.0, xlim, self.mf/self.sf)
        ax1.plot(t, self.abc, label=["a", "b", "c"])
        ax2.plot(t, self.dqz, label=["d", "q", "z"])
        ax3.plot(t, self.theta_ref, label="Theta ref")
        ax3.plot(t, self.theta, label="Theta")
        ax3.plot(t, self.phi, label="Phi")
        #ax3.plot(t, self.omega, label="Omega")

        ax1.legend()
        ax2.legend()
        ax3.legend()
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

    def calculate_alpha_beta_zero(self, a=0.0, b=0.0, c=0.0):
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

    def track(self, step):
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
