# coding=utf-8
import matplotlib.pyplot as plt

from __init__ import *

this_script_root = join(data_root,'Dataframe')

class Gen_Dataframe:
    def __init__(self):
        self.data_dir = join(this_script_root,'')
        self.dff = join(self.data_dir,'HLS_chips.df')
        pass

    def run(self):
        # self.gen_HLS_df()
        df = self.__gen_df_init()
        # df = self.add_band_values(df)
        # df = self.add_indices(df)
        df = self.add_flux_GPP(df)
        # T.print_head_n(df)
        T.save_df(df,self.dff)
        T.df_to_excel(df,self.dff)
        pass

    def gen_HLS_df(self):
        outf = self.dff
        if isfile(outf):
            print('HLS_chips.df already exists')
            pause()
            pause()
            pause()
        satellite_list = ['HLSL30','HLSS30']
        data_dict = {}
        for sat in satellite_list:
            fdir = join(data_root,'HLS/Download_from_GEE',sat,'tif/chips')
            for SITE_ID  in T.listdir(fdir):
                for tif in T.listdir(join(fdir,SITE_ID)):
                    if not tif.endswith('.tif'):
                        continue
                    tif_path = join(fdir,SITE_ID,tif)
                    tif_path_relative = tif_path.replace(this_root,'[PROJECT_ROOT]/')
                    # /Volumes/NVME4T/GPP_ML/data/HLS/Download_from_GEE/HLSL30/tif/chips/AR-CCg/HLS.L30.T20HPF.20180110T135704.v002.AR-CCg.50x50pixels.tif
                    date_str = tif.split('.')[3].split('T')[0]
                    year,mon,day = int(date_str[:4]),int(date_str[4:6]),int(date_str[6:8])
                    date_obj = datetime.datetime(year,mon,day)
                    doy = date_obj - datetime.datetime(year,1,1)
                    doy = doy.days + 1
                    TILE_ID = tif.split('.')[2]

                    data_dict_i = {
                        'tif_path':tif_path_relative,
                        'SITE_ID':SITE_ID,
                        'date':date_obj,
                        'year':year,
                        'mon':mon,
                        'day':day,
                        'doy':doy,
                        'Satellite':sat,
                    }
                    data_dict[tif] = data_dict_i
        df = T.dic_to_df(data_dict,'Chip',col_order=['tif_path','SITE_ID','Satellite','date','year','mon','day','doy'])
        T.save_df(df,outf)
        T.df_to_excel(df,outf)

    def add_band_values(self,df):
        all_dict = {}
        for i,row in tqdm(df.iterrows(),total=len(df)):
            tif_path = row['tif_path']
            # Chip = row['Chip']
            tif_path = tif_path.replace('[PROJECT_ROOT]/',this_root)
            band_list,band_name_list, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array_multiple_bands(tif_path)
            dict_i = {}
            for j in range(len(band_list)):
                arr = band_list[j]
                band_name = band_name_list[j]
                arr_mean = np.nanmean(arr)
                dict_i[band_name] = arr_mean
            all_dict[i] = dict_i
            # if i > 4:
            #     break
        # pprint(all_dict)
        for i,row in tqdm(df.iterrows(),total=len(df)):
            if i not in all_dict:
                continue
            for band in all_dict[i].keys():
                df.at[i,band] = all_dict[i][band]
        return df

    def add_indices(self,df):
        Blue = df['B2'].tolist()
        Green = df['B3'].tolist()
        Red = df['B4'].tolist()
        NIR = df['B5'].tolist()
        SWIR1 = df['B6'].tolist()
        SWIR2 = df['B7'].tolist()

        Blue = np.array(Blue,dtype=np.float32)
        Green = np.array(Green,dtype=np.float32)
        Red = np.array(Red,dtype=np.float32)
        NIR = np.array(NIR,dtype=np.float32)
        SWIR1 = np.array(SWIR1,dtype=np.float32)
        SWIR2 = np.array(SWIR2,dtype=np.float32)

        # test  data
        # Blue = 0.04063388
        # Green = 0.06786016
        # Red = 0.0678922
        # NIR = 0.29526864
        # SWIR1 = 0.23545176
        # SWIR2 = 0.139528

        # test result
        # 0.62610396	0.406648701	-0.000471925	0.112708839	0.184868865	0.373088439

        NDVI = self.cal_NDVI(NIR,Red) # correct
        EVI = self.cal_EVI(NIR,Red,Blue) # correct
        GCI = self.cal_GCI(NIR,Green) # wrong
        NDWI = self.cal_NDWI(SWIR1,NIR) # correct
        NIRv = self.cal_NIRv(NIR,Red) # correct
        kNDVI = self.cal_kNDVI(NIR,Red) # wrong
        # print(NDVI,EVI,GCI,NDWI,NIRv,kNDVI)
        # exit()

        df['NDVI'] = NDVI
        df['EVI'] = EVI
        df['GCI'] = GCI
        df['NDWI'] = NDWI
        df['NIRv'] = NIRv
        df['kNDVI'] = kNDVI
        return df

    def cal_NDVI(self,NIR,RED):
        eps = 1e-6
        return (NIR-RED)/(NIR+RED + eps)

    def cal_EVI(self,NIR,Red,Blue):
        eps = 1e-6
        G, C1, C2, L = 2.5, 6.0, 7.5, 1.0
        EVI = G * (NIR - Red) / (NIR + C1 * Red - C2 * Blue + L)
        return EVI

    def cal_GCI(self,NIR,Green):
        eps = 1e-6
        GCI = (NIR / (Green + eps)) - 1.0
        return GCI

    def cal_NDWI(self,SWIR1,NIR):
        eps = 1e-6
        # NDWI = (Green - NIR) / (Green + NIR + eps)
        NDWI = (NIR - SWIR1) / (NIR + SWIR1 + eps)
        return NDWI

    def cal_NIRv(self,NIR,RED):
        NDVI = self.cal_NDVI(NIR,RED)
        NIRv = NIR * NDVI
        return NIRv

    def cal_kNDVI(self,NIR,Red):
        SIGMA = 0.1
        eps = 1e-6
        a = np.exp(-((NIR - Red) ** 2) / (2.0 * SIGMA ** 2))
        b = np.exp(-((NIR + Red) ** 2) / (2.0 * SIGMA ** 2))
        kNDVI = (a - b) / (a + b + eps)
        return kNDVI

    def add_flux_GPP(self,df):
        flux_dff = join(data_root,'flux_data/dataframe/fluxdata_DD.df')
        flux_df = T.load_df(flux_dff)
        gpp_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            SITE_ID = row['SITE_ID']
            year = row['year']
            month = row['mon']
            day = row['day']
            date_str = f'{year}{month:02d}{day:02d}'
            date_int = int(date_str)
            gpp_df_i = flux_df[flux_df['SITE_ID']==SITE_ID]
            gpp = gpp_df_i[date_int].values[0]
            gpp_list.append(gpp)
        df['GPP'] = gpp_list
        return df

    def __gen_df_init(self):
        if not os.path.isfile(self.dff):
            df = pd.DataFrame()
            T.save_df(df,self.dff)
            return df
        else:
            df,dff = self.__load_df()
            return df

    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        print('len(df):',len(df))
        return df,dff
def main():
    Gen_Dataframe().run()
    pass

if __name__ == '__main__':
    main()