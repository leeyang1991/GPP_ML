# coding=utf-8

import urllib3
from __init__ import *
import ee
import math
import pprint
# coding=utf-8
import geopandas as gpd
from geopy import Point
from geopy.distance import distance as Distance
from shapely.geometry import Polygon

this_script_root = join(data_root,'HLS')

class Expand_points_to_rectangle:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'Expand_points_to_rectangle',
            this_script_root, mode=2)
        pass

    def run(self):
        point_list,name_list = self.read_point_shp()
        rectangle_list = self.expand_points_to_rectangle(point_list)
        self.write_rectangle_shp(rectangle_list,name_list)
        pass

    def read_point_shp(self):

        flux_dff = join(data_root,'flux_data/dataframe/fluxdata_DD.df')
        flux_df = T.load_df(flux_dff)
        site_list = flux_df['SITE_ID'].tolist()
        flux_metadata_dff = join(data_root,'flux_data/metadata/metadata.df')
        flux_metadata_df = T.load_df(flux_metadata_dff)
        T.print_head_n(flux_metadata_df)
        flux_metadata_dic = T.df_to_dic(flux_metadata_df,'SITE_ID')
        lon_list = [flux_metadata_dic[site_id]['LOCATION_LONG'] for site_id in site_list]
        lat_list = [flux_metadata_dic[site_id]['LOCATION_LAT'] for site_id in site_list]
        point_list = zip(lon_list, lat_list)
        point_list = list(point_list)
        name_list = site_list
        return point_list,name_list

    def expand_points_to_rectangle(self, point_list):
        distance_i = 25*30/1000. # km
        # print(point_list)
        rectangle_list = []
        for point in point_list:
            lon = point[0]
            lat = point[1]
            p = Point(latitude=lat, longitude=lon)
            north = Distance(kilometers=distance_i).destination(p, 0)
            south = Distance(kilometers=distance_i).destination(p, 180)
            east = Distance(kilometers=distance_i).destination(p, 90)
            west = Distance(kilometers=distance_i).destination(p, 270)
            # rectangle = Polygon([(west.longitude, west.latitude), (east.longitude, east.latitude),
            #                         (north.longitude, north.latitude), (south.longitude, south.latitude)])
            # east = (east.longitude, east.latitude)
            # west = (west.longitude, west.latitude)
            # north = (north.longitude, north.latitude)
            # south = (south.longitude, south.latitude)

            east_lon = east.longitude
            west_lon = west.longitude
            north_lat = north.latitude
            south_lat = south.latitude

            ll_point = (west_lon, south_lat)
            lr_point = (east_lon, south_lat)
            ur_point = (east_lon, north_lat)
            ul_point = (west_lon, north_lat)

            polygon_geom = Polygon([ll_point, lr_point, ur_point, ul_point])

            rectangle_list.append(polygon_geom)
        return rectangle_list

    def write_rectangle_shp(self, rectangle_list,name_list):
        outdir = join(self.this_class_arr, 'sites')
        T.mkdir(outdir)
        outf = join(outdir, 'sites.shp')
        crs = {'init': 'epsg:4326'}  # 设置坐标系
        polygon = gpd.GeoDataFrame(crs=crs, geometry=rectangle_list)  # 将多边形对象转换为GeoDataFrame对象
        polygon['name'] = name_list

        # 保存为shp文件
        polygon.to_file(outf)
        pass

    def GetDistance(self,lng1, lat1, lng2, lat2):
        radLat1 = self.rad(lat1)
        radLat2 = self.rad(lat2)
        a = radLat1 - radLat2
        b = self.rad(lng1) - self.rad(lng2)
        s = 2 * math.asin(math.sqrt(
            math.pow(math.sin(a / 2), 2) + math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b / 2), 2)))
        s = s * 6378.137 * 1000
        distance = round(s, 4)
        return distance

        pass

    def rad(self,d):
        return d * math.pi / 180


class Download_from_GEE:

    def __init__(self):
        '''
        band: B2, ... , B7
        '''
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'Download_from_GEE',
            this_script_root, mode=2)
        self.collection = 'NASA/HLS/HLSL30/v002'
        # ee.Initialize(project='groovy-bay-461503-r6')
        # ee.Authenticate()
        ee.Initialize(project='lyfq-263413')
        # ee.Initialize()

        # ee.Authenticate()
        # pause()
        # exit()

    def run(self):
        year_list = list(range(2013,2025))
        # self.download_images(2020)
        # MULTIPROCESS(self.download_images,year_list).run(process=10,process_or_thread='t')
        for year in year_list:
            self.download_images(year)
        # self.check()
        # self.unzip()
        # self.reproj()
        # self.clip()
        # self.statistic()
        pass


    def download_images(self,year=1982):
        outdir = join(self.this_class_arr,str(year))
        T.mk_dir(outdir,force=True)
        startDate = f'{year}-01-01'
        endDate = f'{year+1}-01-01'

        rectangle_f = join(Expand_points_to_rectangle().this_class_arr, 'sites/sites.shp')
        rectangle_df = gpd.read_file(rectangle_f)
        geometry_list = rectangle_df['geometry'].tolist()
        site_list = rectangle_df['name'].tolist()

        param_list = []
        for i,geo in enumerate(geometry_list):
            param = (site_list,i,outdir,geo,startDate,endDate)
            param_list.append(param)
            # self.kernel_download_from_gee(param)
        MULTIPROCESS(self.kernel_download_from_gee,param_list).run(process=10,process_or_thread='t')

    def kernel_download_from_gee(self,param):
        site_list,i,outdir,geo,startDate,endDate = param
        site = site_list[i]
        # print(site)
        outdir_i = join(outdir, site)
        T.mk_dir(outdir_i)
        ll = geo.bounds[0:2]
        ur = geo.bounds[2:4]
        region = ee.Geometry.Rectangle(ll[0], ll[1], ur[0], ur[1])

        Collection = ee.ImageCollection(self.collection)
        Collection = Collection.filterDate(startDate, endDate).filterBounds(region)

        info_dict = Collection.getInfo()
        # pprint.pprint(info_dict)
        # print(len(info_dict['features']))
        # exit()
        ids = info_dict['features']
        for i in ids:
            dict_i = eval(str(i))
            # pprint.pprint(dict_i['id'])
            # exit()
            outf_name = dict_i['id'].split('/')[-1] + '.zip'
            out_path = join(outdir_i, outf_name)
            if isfile(out_path):
                continue
            # print(outf_name)
            # exit()
            # print(dict_i['id'])
            # l8 = l8.median()
            # l8_qa = l8.select(['QA_PIXEL'])
            # l8_i = ee.Image(dict_i['LANDSAT/LC08/C02/T1_L2/LC08_145037_20200712'])
            Image = ee.Image(dict_i['id'])
            # Image_product = Image.select('total_precipitation')
            Image_product = Image.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', ])
            # print(Image_product);exit()
            # region = [-111, 32.2, -110, 32.6]# left, bottom, right,
            # region = [-180, -90, 180, 90]  # left, bottom, right,
            exportOptions = {
                'scale': 30,
                'maxPixels': 1e13,
                'region': region,
                # 'fileNamePrefix': 'exampleExport',
                # 'description': 'imageToAssetExample',
            }
            url = Image_product.getDownloadURL(exportOptions)
            # print(url)

            try:
                self.download_i(url, out_path)
            except:
                print('download error', out_path)
                continue
        pass



    def download_i(self,url,outf):
        # try:
        http = urllib3.PoolManager()
        r = http.request('GET', url, preload_content=False)
        body = r.read()
        with open(outf, 'wb') as f:
            f.write(body)

    def unzip(self):
        fdir = join(self.this_class_arr,self.product)
        outdir = join(self.this_class_arr,'unzip',self.product)
        T.mk_dir(outdir,force=True)
        for folder in T.listdir(fdir):
            print(folder)
            fdir_i = join(fdir,folder)
            # T.open_path_and_file(fdir_i,folder)
            # exit()
            outdir_i = join(outdir,folder)
            T.unzip(fdir_i,outdir_i)
        pass

    def check(self):
        fdir = join(self.this_class_arr, self.product)
        # outdir = join(self.this_class_arr, 'unzip', self.product)
        # T.mk_dir(outdir, force=True)
        for folder in T.listdir(fdir):
            fdir_i = join(fdir, folder)
            for f in tqdm(T.listdir(fdir_i),desc=folder):
                fpath = join(fdir_i, f)
                try:
                    zipfile.ZipFile(fpath, 'r')
                except:
                    os.remove(fpath)
                    print(fpath)
                    continue
                pass
        pass


    def wkt(self):
        wkt = '''
        PROJCS["Sinusoidal",
    GEOGCS["GCS_Undefined",
        DATUM["Undefined",
            SPHEROID["User_Defined_Spheroid",6371007.181,0.0]],
        PRIMEM["Greenwich",0.0],
        UNIT["Degree",0.0174532925199433]],
    PROJECTION["Sinusoidal"],
    PARAMETER["False_Easting",0.0],
    PARAMETER["False_Northing",0.0],
    PARAMETER["Central_Meridian",0.0],
    UNIT["Meter",1.0]]'''
        return wkt

    def reproj(self):
        fdir = join(self.this_class_arr,'unzip',self.product)
        outdir = join(self.this_class_arr,'reproj',self.product)
        T.mk_dir(outdir,force=True)
        for year in tqdm(T.listdir(fdir)):
            for date in T.listdir(join(fdir,year)):
                for f in T.listdir(join(fdir,year,date)):
                    fpath = join(fdir,year,date,f)
                    date_str = f.split('.')[0]
                    y,m,d = date_str.split('_')
                    # print(fpath);exit()
                    outpath = join(outdir,f'{y}{m}{d}.tif')
                    # print(outpath)
                    SRS = DIC_and_TIF().gen_srs_from_wkt(self.wkt())
                    wkg_wgs84 = DIC_and_TIF().wkt_84()
                    ToRaster().resample_reproj(fpath,outpath,.5,srcSRS=SRS, dstSRS=wkg_wgs84)
                    # print(outpath)
                    # exit()

    def clip(self):
        fdir = join(self.this_class_arr,'reproj',self.product)
        outdir = join(self.this_class_arr,'clip',self.product)
        #/mnt/sdb2/yang/Global_Resilience/MODIS_download/NDVI_250m_bighorn/arr/bighorn_shp/bighorn_shp
        shp = join(self.this_class_arr,'bighorn_shp','bighorn_shp')
        T.mkdir(outdir,force=True)
        for f in tqdm(T.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir,f)
            outpath = join(outdir,f)

            ToRaster().clip_array(fpath,outpath,shp)
            # T.open_path_and_file(outdir)
            # exit()

        pass

    def statistic(self):
        fdir = join(self.this_class_arr,'reproj',self.product)
        statistic_dict = {}
        for f in T.listdir(fdir):
            date = f.split('.')[0]
            year,mon,day = date[:4],date[4:6],date[6:]
            if not year in statistic_dict:
                statistic_dict[year] = []
            statistic_dict[year].append(f)
        for year in statistic_dict:
            flist = statistic_dict[year]
            print(year,len(flist))


def main():
    # Expand_points_to_rectangle().run()
    Download_from_GEE().run()


    pass


if __name__ == '__main__':
    main()