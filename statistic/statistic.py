# coding=utf-8
import shutil

import matplotlib.pyplot as plt

from __init__ import *
this_script_root = join(results_root,'statistic')

class Train_data_Stat:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'Train_data_Stat',
            this_script_root, mode=2)
        self.dff = join(self.this_class_arr,'dataframe/HLS_chips.df')

    def run(self):
        # self.copy_df()
        self.plot_scatter()
        pass

    def copy_df(self):
        if isfile(self.dff):
            print('dataframe exists')
            pause()
            pause()
            pause()
        outdir = join(self.this_class_arr,'dataframe')
        T.mkdir(outdir)
        origin_dff = join(data_root,'Dataframe/HLS_chips.df')
        shutil.copy(origin_dff,self.dff)
        shutil.copy(origin_dff+'.xlsx',self.dff+'.xlsx')

    def plot_scatter(self):
        df = T.load_df(self.dff)
        T.print_head_n(df)
        GPP_flux = df['GPP'].tolist()
        rs_vals = df['NIRv'].tolist()
        KDE_plot().plot_scatter(GPP_flux,rs_vals)
        plt.show()

        pass

def main():

    Train_data_Stat().run()

if __name__ == '__main__':
    main()