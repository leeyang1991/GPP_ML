# coding=utf-8
from __init__ import *
import ee
import geopandas as gpd

this_script_root = join(data_root,'ERA5')

class Download_from_GEE:

    def __init__(self):
        '''
        band: temperature_2m, temperature_2m_min, temperature_2m_max, total_precipitation_sum,
        https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_DAILY_AGGR
        '''
        # --------------------------------------------------------------------------------
        self.collection = 'ECMWF/ERA5_LAND/DAILY_AGGR'
        # --------------------------------------------------------------------------------
        self.satellite = self.collection.split('/')[2]
        # print(satellite);exit()
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            f'Download_from_GEE/{self.satellite}',
            this_script_root, mode=2)

        # ee.Authenticate()
        # ee.Initialize(project='lyfq-263413')

        # pause()
        # exit()

    def run(self):
        self.extract_time_series_from_GEE()
        pass


    def extract_time_series_from_GEE(self):
        import HLS
        start_year = 2013
        end_year = 2024

        outdir = join(self.this_class_arr,'GEE_download',f'{start_year}-{end_year}')
        T.mk_dir(outdir,force=True)
        startDate = f'{start_year}-01-01'
        endDate = f'{end_year+1}-01-01'

        rectangle_f = join(download_HLS.Expand_points_to_rectangle().this_class_arr, 'sites/sites.shp')
        rectangle_df = gpd.read_file(rectangle_f)
        geometry_list = rectangle_df['geometry'].tolist()
        site_list = rectangle_df['name'].tolist()

        param_list = []
        for i,geo in enumerate(geometry_list):
            param = (site_list,i,outdir,geo,startDate,endDate)
            param_list.append(param)
            # self.kernel_download_from_gee(param)
        MULTIPROCESS(self.kernel_download_from_gee,param_list).run(process=20,process_or_thread='t')

    def kernel_download_from_gee(self,param):
        site_list,i,outdir,geo,startDate,endDate = param
        site = site_list[i]
        ll = geo.bounds[0:2]
        ur = geo.bounds[2:4]
        region = ee.Geometry.Rectangle(ll[0], ll[1], ur[0], ur[1])
        bands = ['temperature_2m','temperature_2m_min','temperature_2m_max','total_precipitation_sum']

        def extract_time_series(img):
            values = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=10000,  # ERA5 分辨率约 9km，10km 足够
                bestEffort=True
            )
            return ee.Feature(None, {
                "date": img.date().format("YYYY-MM-dd"),
                **{b: values.get(b) for b in bands}
            })

        Collection = ee.ImageCollection(self.collection)
        dataset = Collection.filterDate(startDate, endDate).filterBounds(region).select(bands)
        features = dataset.map(extract_time_series).getInfo()
        records = [f["properties"] for f in features["features"]]
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        outf = join(outdir,f'{site}.df')
        T.save_df(df,outf)
        T.df_to_excel(df,outf)


def main():
    Download_from_GEE().run()
    pass


if __name__ == '__main__':
    main()