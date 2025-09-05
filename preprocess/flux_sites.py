
from __init__ import *

class Fluxdata_summary:

    def __init__(self):
        self.data_dir = join(data_root, 'flux_data/')
        pass

    def run(self):
        # metadata_df = self.get_metadata_from_xlsx()
        # self.plot_spatial()
        # self.plot_time_ranges()
        self.get_col_names()

        pass

    def get_metadata_from_xlsx(self):

        outdir = join(self.data_dir, 'metadata')
        outf = join(outdir, 'metadata.df')
        if isfile(outf):
            df = T.load_df(outf)
            return df
        fpath = join(self.data_dir, 'metadata/AMF_AA-Net_BIF_CCBY4_20250201.xlsx')
        df = pd.read_excel(fpath)
        site_group_dict = T.df_groupby(df, 'SITE_ID')
        all_site_dict = {}
        selected_variable_list = self.selected_columns()
        for site in site_group_dict:
            site_df = site_group_dict[site]
            VARIABLE_list = site_df['VARIABLE'].tolist()
            DATAVALUE_list = site_df['DATAVALUE'].tolist()
            site_dict = dict(zip(VARIABLE_list, DATAVALUE_list))
            selected_site_dict = {k:v for k,v in site_dict.items() if k in selected_variable_list}
            all_site_dict[site] = selected_site_dict
        all_site_df = T.dic_to_df(all_site_dict, key_col_str='SITE_ID')

        all_site_df['FLUX_MEASUREMENTS_DATE_START'] = all_site_df['FLUX_MEASUREMENTS_DATE_START'].apply(lambda x: str(int(x)) if pd.notnull(x) else '')
        all_site_df['FLUX_MEASUREMENTS_DATE_END'] = all_site_df['FLUX_MEASUREMENTS_DATE_END'].apply(lambda x: str(int(x)) if pd.notnull(x) else '')

        T.print_head_n(all_site_df)
        T.save_df(all_site_df, outf)
        T.df_to_excel(all_site_df, outf)
        return all_site_df
        # example_site = all_site_df.iloc[0]
        # cols = all_site_df.columns.tolist()
        # example_dict = {}
        # for col in cols:
        #     # example_dict[col] = example_site[col]
        #     val = example_site[col]
        #     # print(type(val))
        #     try:
        #         if np.isnan(val):
        #             continue
        #     except:
        #         pass
        #     print(col,'\t'*5, example_site[col])
        #     print('-------------')

    def selected_columns(self):
        variable_list_str = '''
        FLUX_MEASUREMENTS_DATE_END
        FLUX_MEASUREMENTS_DATE_START
        IGBP
        LOCATION_ELEV
        LOCATION_LAT
        LOCATION_LONG
        MAP
        MAT
        '''
        variable_list = variable_list_str.split('\n')
        variable_list = [i.strip() for i in variable_list]
        return variable_list

    def plot_spatial(self):
        metadata_df = self.get_metadata_from_xlsx()
        fdir = join(self.data_dir,'csv')
        site_name_list = []
        for folder in T.listdir(fdir):
            site_name = folder.split('_')[1]
            if not site_name in site_name_list:
                site_name_list.append(site_name)
            else:
                raise ValueError(f'duplicated site name {site_name}')
        print(len(site_name_list))
        metadata_dict = T.df_to_dic(metadata_df, key_str='SITE_ID')
        lon_list = []
        lat_list = []
        for site_name in site_name_list:
            site_dict = metadata_dict[site_name]
            lat = site_dict['LOCATION_LAT']
            if lat < 0:
                continue
            lon = site_dict['LOCATION_LONG']
            lon_list.append(lon)
            lat_list.append(lat)
        DIC_and_TIF().plot_sites_location(lon_list, lat_list, background_tif=global_land_tif, inshp=None, out_background_tif=None,
                        pixel_size=None, text_list=None, colorlist=None, isshow=False,s=2,color='k')
        plt.show()
        print(min(lat_list))

        pass

    def plot_time_ranges(self):
        fdir = join(self.data_dir, 'csv')
        site_name_list = []
        site_date_dict = {}
        for folder in tqdm(T.listdir(fdir)):
            site_name = folder.split('_')[1]
            if not site_name in site_name_list:
                site_name_list.append(site_name)
            else:
                raise ValueError(f'duplicated site name {site_name}')
            folder_i = join(fdir, folder)
            fname_split = folder.split('_')
            fname_split.insert(4,'MM')
            fname = '_'.join(fname_split) + '.csv'
            fname = fname.replace('(1)','')
            fname = fname.replace(' ','')
            fpath = join(folder_i, fname)
            df_i = pd.read_csv(fpath)
            TIMESTAMP = df_i['TIMESTAMP'].tolist()
            TIMESTAMP_str = [str(i) for i in TIMESTAMP]
            TIMESTAMP_obj_list = []
            for ts in TIMESTAMP_str:
                year = int(ts[:4])
                month = int(ts[4:6])
                date_obj = datetime.datetime(year, month, 1)
                TIMESTAMP_obj_list.append(date_obj)
            start_date = min(TIMESTAMP_obj_list)
            end_date = max(TIMESTAMP_obj_list)
            delta = end_date - start_date
            dict_i = {
                'start_date':start_date,
                'end_date':end_date,
                'delta':delta
            }
            site_date_dict[site_name] = dict_i
        df_site_date = T.dic_to_df(site_date_dict, key_col_str='site_name')
        df_site_date_rank = df_site_date.sort_values(by='delta', ascending=True, ignore_index=True)
        T.print_head_n(df_site_date_rank)
        plt.figure(figsize=(10, 40))

        for i,row in df_site_date_rank.iterrows():
            site_name = row['site_name']
            start_date = row['start_date']
            end_date = row['end_date']
            plt.plot([start_date, end_date], [i, i], lw=2, color='k')
        plt.yticks(range(len(df_site_date_rank)), df_site_date_rank['site_name'].tolist())
        plt.grid(True)
        plt.show()

    def get_col_names(self):
        fdir = join(self.data_dir, 'csv')
        site_name_list = []
        cols_list = []
        for folder in tqdm(T.listdir(fdir)):
            site_name = folder.split('_')[1]
            if not site_name in site_name_list:
                site_name_list.append(site_name)
            else:
                raise ValueError(f'duplicated site name {site_name}')
            folder_i = join(fdir, folder)
            fname_split = folder.split('_')
            fname_split.insert(4,'MM')
            fname = '_'.join(fname_split) + '.csv'
            fname = fname.replace('(1)','')
            fname = fname.replace(' ','')
            fpath = join(folder_i, fname)
            df_i = pd.read_csv(fpath)
            cols = df_i.columns.tolist()
            for c in cols:
                if not c in cols_list:
                    cols_list.append(c)
        for c in cols_list:
            print(c)
        pass

class Fluxdata:

    def __init__(self):
        # gpp_var: GPP_NT_VUT_REF
        self.data_dir = join(data_root, 'flux_data/')
        self.temporal_res = 'DD'
        pass

    def run(self):
        # self.gen_dataframe()
        self.plot_time_series()
        # self.plot_HANTS_interpolation()
        # self.HANTS_interpolation()
        # self.plot_clean_HANTS_interpolation()
        # self.clean_HANTS_interpolation()

        pass

    def gen_dataframe(self):
        fdir = join(self.data_dir, 'csv')
        outdir = join(self.data_dir, 'dataframe')
        T.mk_dir(outdir)
        all_dict = {}
        for folder in tqdm(T.listdir(fdir)):
            folder = str(folder)
            fdir_i = join(fdir, folder)
            fname_split = folder.split('_')
            fname_split.insert(4, self.temporal_res)
            fname = '_'.join(fname_split) + '.csv'
            fname = fname.replace('(1)', '')
            fname = fname.replace(' ', '')
            fpath = join(fdir_i, fname)
            df_i = pd.read_csv(fpath)
            site_name = folder.split('_')[1]
            TIMESTAMP = df_i['TIMESTAMP'].tolist()
            GPP = df_i['GPP_NT_VUT_REF'].tolist()
            QC = df_i['NEE_VUT_REF_QC'].tolist()
            QC = np.array(QC,dtype=float)
            GPP = np.array(GPP,dtype=np.float32)
            GPP[GPP<0] = np.nan
            GPP[QC<0.6] = np.nan
            dict_i = T.dict_zip(TIMESTAMP, GPP)
            all_dict[site_name] = dict_i
            # break
        df = T.dic_to_df(all_dict, key_col_str='SITE_ID')
        outf = join(outdir, f'fluxdata_{self.temporal_res}.df')
        T.save_df(df, outf)
        T.df_to_excel(df,outf)

        pass

    def plot_time_series(self):
        fdir = join(self.data_dir, 'dataframe')
        df = T.load_df(join(fdir, 'fluxdata_DD.df'))
        date_str_list = df.columns.tolist()
        date_str_list.remove('SITE_ID')
        # print(date_str_list)
        date_obj_list = []
        for date_str in date_str_list:
            date_str = str(date_str)
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            date_obj = datetime.datetime(year, month, day)
            date_obj_list.append(date_obj)
        plt.figure(figsize=(16, 4))
        for i,row in df.iterrows():
            site_name = row['SITE_ID']
            vals = row[date_str_list].tolist()
            plt.plot(date_obj_list, vals,label=site_name)
            plt.legend()
            plt.tight_layout()
            plt.show()
        pass

    def HANTS_interpolation(self):
        outdir = join(self.data_dir, 'HANTS_interpolation')
        T.mk_dir(outdir)
        fdir = join(self.data_dir, 'dataframe')
        df = T.load_df(join(fdir, 'fluxdata_DD.df'))
        date_str_list = df.columns.tolist()
        date_str_list.remove('SITE_ID')
        # print(date_str_list)
        date_obj_list = []
        for date_str in date_str_list:
            date_str = str(date_str)
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            date_obj = datetime.datetime(year, month, day)
            date_obj_list.append(date_obj)
        date_obj_list = np.array(date_obj_list)
        hants_dict_all = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            site_name = row['SITE_ID']
            vals = row[date_str_list].tolist()
            vals = np.array(vals, dtype=np.float32)
            vals_clean = vals[~np.isnan(vals)]
            date_obj_list_clean = date_obj_list[~np.isnan(vals)]
            hants_dict = HANTS().hants_interpolate(vals_clean, date_obj_list_clean, (0,100), nan_value=0,silent=False,valid_ratio=0.5)
            hants_dict_all[site_name] = hants_dict
        outf = join(outdir, 'HANTS_interpolation.npy')
        T.save_npy(hants_dict_all, outf)
        pass

    def plot_clean_HANTS_interpolation(self):
        '''
        delete abnormal year in flux gpp data
        :return:
        '''
        outdir = join(self.data_dir, 'plot_clean_HANTS_interpolation')
        T.mk_dir(outdir)
        fdir = join(self.data_dir, 'HANTS_interpolation')
        hants_dict_all = T.load_npy(join(fdir, 'HANTS_interpolation.npy'))
        for site in tqdm(hants_dict_all):
            hants_dict = hants_dict_all[site]
            max_val_list = []
            for year in hants_dict:
                vals = hants_dict[year]
                max_val = np.nanmax(vals)
                max_val_list.append(max_val)
            median_max_val = np.nanmedian(max_val_list)
            median_max_val = float(median_max_val)
            threshold = 0.5 * median_max_val

            invalid_year_list = []
            plt.figure(figsize=(16, 4))
            for year in hants_dict:
                vals = hants_dict[year]
                max_val = np.nanmax(vals)
                if max_val < threshold:
                    invalid_year_list.append(year)
                base_date = datetime.datetime(year, 1, 1)
                date_obj_list = []
                for i in range(365):
                    date_obj = base_date + datetime.timedelta(days=i)
                    date_obj_list.append(date_obj)
                date_obj_list = np.array(date_obj_list)
                if year in invalid_year_list:
                    plt.plot(date_obj_list, vals,c='r',lw=4)
                else:
                    plt.plot(date_obj_list, vals,c='k')
            if len(invalid_year_list) == 0:
                plt.close()
                continue
            site = str(site)
            invalid_year_list_str = [str(i) for i in invalid_year_list]
            title = site + ':' + ';'.join(invalid_year_list_str)
            plt.title(title)
            plt.tight_layout()
            # plt.show()
            outf = join(outdir, site + '.png')
            plt.savefig(outf)
            plt.close()

    def clean_HANTS_interpolation(self):
        '''
        delete abnormal year in flux gpp data
        :return:
        '''
        threshold = 0.5
        outdir = join(self.data_dir, 'clean_HANTS_interpolation')
        T.mk_dir(outdir)
        fdir = join(self.data_dir, 'HANTS_interpolation')
        hants_dict_all = T.load_npy(join(fdir, 'HANTS_interpolation.npy'))
        invalid_year_dict = {}
        for site in tqdm(hants_dict_all):
            hants_dict = hants_dict_all[site]
            max_val_list = []
            for year in hants_dict:
                vals = hants_dict[year]
                max_val = np.nanmax(vals)
                max_val_list.append(max_val)
            median_max_val = np.nanmedian(max_val_list)
            median_max_val = float(median_max_val)
            val_threshold = threshold * median_max_val

            invalid_year_list = []
            for year in hants_dict:
                vals = hants_dict[year]
                max_val = np.nanmax(vals)
                if max_val < val_threshold:
                    invalid_year_list.append(year)
            invalid_year_dict[site] = invalid_year_list
        for site in hants_dict_all:
            hants_dict = hants_dict_all[site]
            for year in invalid_year_dict[site]:
                del hants_dict[year]
        outf = join(outdir, 'clean_HANTS_interpolation.npy')
        T.save_npy(hants_dict_all, outf)

    def plot_HANTS_interpolation(self):
        outdir = join(self.data_dir, 'plot_HANTS_interpolation')
        T.mk_dir(outdir)
        fdir = join(self.data_dir, 'dataframe')
        df = T.load_df(join(fdir, 'fluxdata_DD.df'))
        date_str_list = df.columns.tolist()
        date_str_list.remove('SITE_ID')
        # print(date_str_list)
        date_obj_list = []
        for date_str in date_str_list:
            date_str = str(date_str)
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            date_obj = datetime.datetime(year, month, day)
            date_obj_list.append(date_obj)
        date_obj_list = np.array(date_obj_list)
        for i, row in tqdm(df.iterrows(), total=len(df)):
            site_name = row['SITE_ID']
            vals = row[date_str_list].tolist()
            vals = np.array(vals, dtype=np.float32)
            vals_clean = vals[~np.isnan(vals)]
            date_obj_list_clean = date_obj_list[~np.isnan(vals)]
            hants_dict = HANTS().hants_interpolate(vals_clean, date_obj_list_clean, (0,100), nan_value=0,silent=False,valid_ratio=0.5)
            # print('---'*8)
            plt.figure(figsize=(16, 4))
            for year in hants_dict:
                # print(year)
                base_date = datetime.datetime(year, 1, 1)
                date_obj_list_i = []
                for i in range(365):
                    date_obj_list_i.append(base_date + datetime.timedelta(days=i))
                vals_hants = hants_dict[year]
                vals_hants = np.array(vals_hants, dtype=np.float32)
                plt.plot(date_obj_list_i, vals_hants,color='r',zorder=1)
            plt.plot(date_obj_list_clean, vals_clean, label='original', alpha=0.5,color='k',zorder=-1)
            plt.legend()
            plt.title(site_name)
            plt.tight_layout()
            plt.show()
            # outf = join(outdir, f'{site_name}.png')
            # plt.savefig(outf,dpi=144)
            # plt.close()

        pass


def main():
    Fluxdata_summary().run()

if __name__ == '__main__':
    main()