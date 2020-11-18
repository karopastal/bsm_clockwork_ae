from datetime import datetime

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import v8.WT as WT
import v8.SBLD as SB
import v8.LSign as LSign

BASEDIR = 'data/cwt_generator'
PATH_SIGNALS_CATALOG = BASEDIR + '/signals_catalog'


class CWTGenerator:
    def __init__(self,
                 min_mass=150,
                 max_mass=2700,
                 n_bins=1275,
                 merge_x=20,
                 merge_y=2,
                 cone=False,
                 n_cone_r=2,
                 trials=2000,
                 binned_shape=(60, 56)):

        self.created = False
        self.min_mass = min_mass
        self.max_mass = max_mass
        self.n_bins = n_bins
        self.merge_x = merge_x
        self.merge_y = merge_y
        self.cone = cone
        self.n_cone_r = n_cone_r
        self.trials = trials
        self.binned_shape = binned_shape
        self.fix_background = None
        self.fix_signal = None
        self.w_min_grid = None
        self.w_max_grid = None
        self.p_value_grid = None

    def _create_p_value_grid(self):
        toy_exp_grid, _ = LSign.ToyExperimentGridMaker(self.fix_background,
                                                       self.merge_x,
                                                       self.merge_y,
                                                       self.trials)

        return LSign.pvalueGridMaker(toy_exp_grid)

    def _generate_p_value_grid_from_background(self):
        self.mass_list = np.linspace(self.min_mass, self.max_mass, self.n_bins)
        self.fix_background = SB.Bfix(self.mass_list)

        self.w_min_grid, self.w_max_grid, self.p_value_grid = \
            self._create_p_value_grid()

    def _p_value_wavelet_transform(self, events):
        cwt, _, _ = WT.WaveletTransform(events,
                                        self.merge_x,
                                        self.merge_y,
                                        self.cone,
                                        self.n_cone_r)

        cwt_avg = LSign.pvalueCalc(self.w_min_grid,
                                   self.w_max_grid,
                                   self.p_value_grid,
                                   cwt)

        return -1*np.log(cwt_avg)

    def _serialize_config(self):
        config = dict()

        config['min_mass'] = self.min_mass
        config['max_mass'] = self.max_mass
        config['n_bins'] = self.n_bins
        config['merge_x'] = self.merge_x
        config['merge_y'] = self.merge_y
        config['cone'] = self.cone
        config['n_cone_r'] = self.n_cone_r
        config['trials'] = self.trials
        config['mass_list'] = self.mass_list
        config['fix_background'] = self.fix_background
        config['w_min_grid'] = self.w_min_grid
        config['w_max_grid'] = self.w_max_grid
        config['p_value_grid'] = self.p_value_grid
        config['binned_shape'] = self.binned_shape

        return config

    def create(self, name):
        self._generate_p_value_grid_from_background()
        config = self._serialize_config()
        config['name'] = name

        path = BASEDIR + "/" + name
        os.makedirs(path, exist_ok=True)
        timestamp = datetime.now().strftime("%m-%d-%yT%H-%M-%S")
        filename = path + "/" + timestamp
        outfile = open(filename, 'wb')
        pickle.dump(config, outfile)
        outfile.close()

        self.created = True

        return config

    def load(self, path):
        infile = open(path, 'rb')
        config = pickle.load(infile)

        for key, value in config.items():
            setattr(self, key, value)

        infile.close()
        self.created = True

    def mount_signal(self, m_5, k):
        path = PATH_SIGNALS_CATALOG + '/m_5_%s__k_%s' % (m_5, k,)

        if os.path.isfile(path):
            infile = open(path, 'rb')
            self.fix_signal = pickle.load(infile)
            infile.close()
        else:
            self.fix_signal, _ = SB.Sfix(m_5, k, self.mass_list)

            os.makedirs(PATH_SIGNALS_CATALOG, exist_ok=True)
            outfile = open(path, 'wb')
            pickle.dump(self.fix_signal, outfile)
            outfile.close()

    def yield_background(self):
        if not self.created:
            return "Error: use .create(name='') or .load(path='')"

        events = SB.SBmain(self.fix_background,
                           self.fix_background,
                           aB=1,
                           aS=0)

        cwt_p_value = self._p_value_wavelet_transform(events)

        return cwt_p_value[0:self.binned_shape[0], 0:self.binned_shape[1]]

    def yield_background_signal(self):
        if not self.created:
            print("Error: use .create(name='') or .load(path='')")
        if not self.fix_signal:
            print("Error: use mount_signal(m_5={int}, k={int})")

        events = SB.SBmain(self.fix_background,
                           self.fix_signal,
                           aB=1,
                           aS=1)

        cwt_p_value = self._p_value_wavelet_transform(events)

        return cwt_p_value[0:self.binned_shape[0], 0:self.binned_shape[1]]

    def plot(self, wt_p_value):
        fig, ax = plt.subplots(figsize=(9, 7))

        img = ax.imshow(wt_p_value,
                        extent=(self.min_mass, self.max_mass, self.binned_shape[1], 0),
                        interpolation='sinc',
                        aspect='auto',
                        cmap='bwr')

        ax.set_ylim(0, 56)

        cbar = fig.colorbar(img, ax=ax)
        cbar.ax.tick_params(labelsize=18)

        plt.title('CWT of Background + Signal w/ Fluctuations - local p-value', fontsize=19)
        plt.ylabel('Scales', fontsize=18)
        plt.xlabel('Mass', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        plt.show()


# cwt_generator = CWTGenerator()
# cwt_generator.load(path='data/cwt_generator/type_1/11-16-20T12-36-42')
#
# cwt_generator.mount_signal(m_5=7000, k=750)
# signal = cwt_generator.yield_background_signal()
#
# cwt_generator.plot(signal)
