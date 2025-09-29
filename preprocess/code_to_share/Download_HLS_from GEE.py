# coding=utf-8
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=UserWarning)
warnings.filterwarnings("ignore",category=FutureWarning)

import sys
python_version = sys.version_info.major
assert python_version == 3, 'Python Version Error'

from lytools import * # This is the package I released, pip install lytools, need to configure the gdal environment before installing lytools.
T = Tools()
import ee
import math
import geopandas as gpd
from geopy import Point
from geopy.distance import distance as Distance
from shapely.geometry import Polygon
import urllib3
import shutil

this_root = 'project_files/'
data_root = join(this_root,'data')
this_script_root = join(this_root,'HLS')

class Expand_points_to_rectangle:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Expand_points_to_rectangle',this_script_root, mode=2)
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
        crs = {'init': 'epsg:4326'}  # Set coordinate system
        polygon = gpd.GeoDataFrame(crs=crs, geometry=rectangle_list)  # Convert polygon objects to GeoDataFrame objects
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
        band: B2, ... , B7, fmask
        https://developers.google.com/earth-engine/datasets/catalog/NASA_HLS_HLSS30_v002
        https://developers.google.com/earth-engine/datasets/catalog/NASA_HLS_HLSL30_v002
        '''
        # --------------------------------------------------------------------------------
        # self.collection = 'NASA/HLS/HLSL30/v002' # derived from LandSAT
        # --------------------------------------------------------------------------------
        self.collection = 'NASA/HLS/HLSS30/v002' # derived from Sentinel
        # --------------------------------------------------------------------------------
        self.satellite = self.collection.split('/')[2]
        # print(satellite);exit()
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            f'Download_from_GEE/{self.satellite}',
            this_script_root, mode=2)

        # ee.Authenticate()
        ee.Initialize(project='lyfq-263413')


    def run(self):
        year_list = list(range(2016,2025))
        for year in year_list:
            self.download_images(year)
        self.check()
        self.unzip()
        self.resize_to_50_x_50()
        self.quality_control()
        self.fill_nan_values() # fill nan value with 0 [optional]
        self.merge_bands()
        self.pick_images_based_on_flux_site_and_rename()

        pass


    def download_images(self,year):
        outdir = join(self.this_class_arr,'GEE_download',str(year))
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
            # self.kernel_download_from_gee(param) ##### download one by one, slow, for debug use
        ##### download 20 sites in parallel, fast
        MULTIPROCESS(self.kernel_download_from_gee,param_list).run(process=20,process_or_thread='t',desc=f'download_{year}')

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
            Image = ee.Image(dict_i['id'])
            Image_product = Image.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'Fmask'])
            exportOptions = {
                'scale': 30,
                'maxPixels': 1e13,
                'region': region,
            }
            url = Image_product.getDownloadURL(exportOptions)

            try:
                self.download_i(url, out_path)
            except Exception as e:
                print(e, 'download error', out_path)
                continue
        pass


    def download_i(self,url,outf):
        http = urllib3.PoolManager()
        r = http.request('GET', url, preload_content=False)
        body = r.read()
        with open(outf, 'wb') as f:
            f.write(body)


    def unzip(self):
        fdir = join(self.this_class_arr,'GEE_download')
        outdir = join(self.this_class_arr,'unzip')
        T.mk_dir(outdir,force=True)
        for folder in T.listdir(fdir):
            print(folder)
            fdir_i = join(fdir,folder)
            outdir_i = join(outdir,folder)
            T.mkdir(outdir_i)
            for site in T.listdir(join(fdir_i)):
                fdir_ii = join(fdir_i,site)
                outdir_ii = join(outdir_i,site)
                T.mkdir(outdir_ii)
                T.unzip(fdir_ii,outdir_ii)
        pass

    def check(self):
        fdir = join(self.this_class_arr,'GEE_download')
        for year in T.listdir(fdir):
            for site in tqdm(T.listdir(join(fdir,year)),desc=f'checking {year}'):
                for f in T.listdir(join(fdir,year,site)):
                    fpath = join(fdir,year,site,f)
                    try:
                        zipfile.ZipFile(fpath, 'r')
                    except:
                        # os.remove(fpath)
                        print(fpath,'error')
                        continue
                    pass
        pass

    def resize_to_50_x_50(self):
        fdir = join(self.this_class_arr,'unzip')
        outdir = join(self.this_class_arr,'resize_to_50_x_50')
        for year in T.listdir(fdir):
            year = str(year)
            for site in tqdm(T.listdir(join(fdir,year)),desc=f'resizing to 50x50, {year}'):
                for date in T.listdir(join(fdir,year,site)):
                    outdir_i = join(outdir,year,site,date)
                    T.mkdir(outdir_i,force=True)
                    for f in T.listdir(join(fdir,year,site,date)):
                        if not f.endswith('.tif'):
                            continue
                        fpath = join(fdir,year,site,date,f)
                        outf = join(outdir_i,f)
                        if isfile(outf):
                            continue
                        array, originX, originY, pixelWidth, pixelHeight,projection_wkt = self.raster2array(fpath)
                        array = np.array(array,dtype=np.float32)
                        # print(array);exit()
                        array[array<=0] = np.nan
                        r_num = array.shape[0]
                        c_num = array.shape[1]
                        if r_num < 50:
                            raise
                        if c_num < 50:
                            raise
                        array = array[:50,:50]
                        self.array2raster(outf, originX, originY, pixelWidth, pixelHeight, array,projection_wkt)
        pass

    def quality_control(self):
        '''
        see:https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf#page=17.08
        fmask value:
        clean pixel values for No water/snow_ice/cloud/cloud_shadow/Adjacent_to_cloud:
        [0,64,128,192]
        bit num|mask name        |bit value|mask description
        7-6    |aerosol level    |11       |High aerosol
        7-6    |aerosol level    |10       |Moderate aerosol
        7-6    |aerosol level    |01       |Low aerosol
        7-6    |aerosol level    |00       |Climatology aerosol
        5      |Water            |1        |Climatology aerosol
        5      |Water            |0        |Climatology aerosol
        4      |Snow/ice         |1        |Yes
        4      |Snow/ice         |0        |No
        3      |Cloud shadow     |1        |Yes
        3      |Cloud shadow     |0        |No
        2      |Adjacent to cloud|1        |Yes
        2      |Adjacent to cloud|0        |No
        1      |Cloud            |1        |Yes
        1      |Cloud            |0        |No
        0      |Cirrus           |NA       |NA
        '''
        data_dir = join(self.this_class_arr,'resize_to_50_x_50')
        outdir = join(self.this_class_arr,'resize_to_50_x_50_after_qc')
        arr_init = np.ones((50,50),dtype=np.uint8)
        arr0 = arr_init * 0
        arr64 = arr_init * 64
        arr128 = arr_init * 128
        arr192 = arr_init * 192

        for year in T.listdir(data_dir):
            for site in tqdm(T.listdir(join(data_dir,year)),desc=f'quality control {year}'):
                for date in T.listdir(join(data_dir,year,site)):
                    outdir_i = join(outdir,year,site,date)
                    T.mkdir(outdir_i,force=True)
                    fpath_qc = join(data_dir,year,site,date,date+'.Fmask.tif')
                    arr_qc = self.raster2array(fpath_qc)[0]
                    arr_qc = np.array(arr_qc,dtype=np.uint8)
                    arr_filter0 = arr_qc==arr0
                    arr_filter64 = arr_qc==arr64
                    arr_filter128 = arr_qc==arr128
                    arr_filter192 = arr_qc==arr192
                    arr_qc_filter = arr_filter0 | arr_filter64 | arr_filter128 | arr_filter192
                    if not isdir(join(data_dir,year,site,date)):
                        continue
                    for f in T.listdir(join(data_dir,year,site,date)):
                        if not f.endswith('.tif'):
                            continue
                        outf = join(outdir_i, f)
                        if isfile(outf):
                            continue
                        if f.endswith('.Fmask.tif'):
                            continue
                        fpath = join(data_dir,year,site,date,f)
                        array, originX, originY, pixelWidth, pixelHeight,projection_wkt = self.raster2array(fpath)
                        array[~arr_qc_filter] = np.nan
                        self.array2raster(outf, originX, originY, pixelWidth, pixelHeight, array,projection_wkt)

        pass

    def merge_bands(self):
        fdir = join(self.this_class_arr,'resize_to_50_x_50_after_qc_fill_nan')
        outdir = join(self.this_class_arr,'merge_bands')
        for year in T.listdir(fdir):
            for site in tqdm(T.listdir(join(fdir,year)),desc=f'merging bands, {year}'):
                outdir_i = join(outdir, year, site)
                T.mkdir(outdir_i, force=True)
                for date in T.listdir(join(fdir,year,site)):
                    outf = join(outdir_i,date + '.tif')
                    tif_list = []
                    for f in T.listdir(join(fdir,year,site,date)):
                        if not f.endswith('.tif'):
                            continue
                        fpath = join(fdir,year,site,date,f)
                        # print(fpath)
                        array, originX, originY, pixelWidth, pixelHeight,projection_wkt = self.raster2array(fpath)
                        array = np.array(array,dtype=np.float32)
                        array_valid = array[~np.isnan(array)]
                        valid_ratio = len(array_valid)/array.size
                        if valid_ratio < 0.5:
                            tif_list = []
                            break
                        tif_list.append(fpath)
                    if len(tif_list) == 0:
                        continue
                    band_name_list = []
                    for f in tif_list:
                        band_name = f.split('.')[-2]
                        band_name_list.append(band_name)
                    self.gdal_merge_bands(tif_list,band_name_list,outf)
                # exit()


    def pick_images_based_on_flux_site_and_rename(self):
        image_dir = join(self.this_class_arr,'merge_bands')
        flux_dff = join(data_root,'flux_data/dataframe/fluxdata_DD.df')
        flux_df = T.load_df(flux_dff)
        outdir = join(self.this_class_tif,'chips')
        T.mkdir(outdir)
        date_list = flux_df.columns.tolist()
        date_list.remove('SITE_ID')
        date_list.sort()
        date_obj_list = []
        for date_str in date_list:
            date_str = str(date_str)
            year,mon,day = int(date_str[:4]),int(date_str[4:6]),int(date_str[6:8])
            date_obj = datetime.datetime(year,mon,day)
            date_obj_list.append(date_obj)
        for i,row in tqdm(flux_df.iterrows(),total=len(flux_df)):
            site_ID = row['SITE_ID']
            outdir_i = join(outdir,site_ID)
            T.mkdir(outdir_i)
            valid_date_for_flux = []
            for i,date in enumerate(date_list):
                val = row[date]
                if np.isnan(val):
                    continue
                valid_date_for_flux.append(date_obj_list[i])
            valid_date_for_HLS = []
            HLS_fpath_date_dict = {}
            for folder in T.listdir(image_dir):
                fdir_i = join(image_dir,folder,site_ID)
                if not isdir(fdir_i):
                    continue
                for f in T.listdir(fdir_i):
                    fpath = join(fdir_i,f)
                    date_str = f.split('_')[1].split('T')[0]
                    year,mon,day = int(date_str[:4]),int(date_str[4:6]),int(date_str[6:8])
                    date_obj = datetime.datetime(year,mon,day)
                    valid_date_for_HLS.append(date_obj)
                    HLS_fpath_date_dict[date_obj] = fpath
            intersect_date = list(set(valid_date_for_HLS) & set(valid_date_for_flux))
            selected_fpath_list = []
            for date_obj in intersect_date:
                selected_fpath_list.append(HLS_fpath_date_dict[date_obj])
            # old name: T19FFV_20140306T135047.tif
            # target name: HLS.S30.T31UES.2020111T105031.v2.0.BE-Lcr_merged.50x50pixels.tif
            # rename:   HLS.L30.T20HPF.20231015T135713.v002.AR-CCg.50x50pixels.tif
            selected_fpath_rename_list = []
            satellite = self.satellite.replace('HLS', '')
            for fpath in selected_fpath_list:
                f_name = fpath.split('/')[-1].split('.')[0]
                tile = f_name.split('_')[0]
                date_str = f_name.split('_')[1]
                new_name = f'HLS.{satellite}.{tile}.{date_str}.v002.{site_ID}.50x50pixels.tif'
                outf = join(outdir_i,new_name)
                shutil.copy(fpath,outf)

        # remove empty folder
        folders = list(os.walk(outdir))[1:]
        for folder in folders:
            if not folder[2]:
                os.rmdir(folder[0])

    def fill_nan_values(self):
        tif_dir = join(self.this_class_arr,'resize_to_50_x_50_after_qc')
        outdir = join(self.this_class_arr,'resize_to_50_x_50_after_qc_fill_nan')
        for year in T.listdir(tif_dir):
            for site in tqdm(T.listdir(join(tif_dir,year)),desc=year):
                for date in T.listdir(join(tif_dir,year,site)):
                    outdir_i = join(outdir, year, site, date)
                    T.mkdir(outdir_i,force=True)
                    for f in T.listdir(join(tif_dir,year,site,date)):
                        if not f.endswith('.tif'):
                            continue
                        fpath = join(tif_dir,year,site,date,f)
                        array, originX, originY, pixelWidth, pixelHeight,projection_wkt = self.raster2array(fpath)
                        array[np.isnan(array)] = 0
                        outf = join(outdir_i,f)
                        self.array2raster(outf, originX, originY, pixelWidth, pixelHeight, array, projection_wkt)
        pass

    def raster2array(self, rasterfn):
        '''
        create array from raster
        Agrs:
            rasterfn: tiff file path
        Returns:
            array: tiff data, an 2D array
        '''
        raster = gdal.Open(rasterfn)
        projection_wkt = raster.GetProjection()
        geotransform = raster.GetGeoTransform()
        originX = geotransform[0]
        originY = geotransform[3]
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[5]
        band = raster.GetRasterBand(1)
        array = band.ReadAsArray()
        array = np.asarray(array)
        del raster
        return array, originX, originY, pixelWidth, pixelHeight,projection_wkt

    def array2raster(self, newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, array, projection_wkt,ndv=-999999):
        cols = array.shape[1]
        rows = array.shape[0]
        originX = longitude_start
        originY = latitude_start
        # open geotiff
        driver = gdal.GetDriverByName('GTiff')
        if os.path.exists(newRasterfn):
            os.remove(newRasterfn)
        outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
        # outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_UInt16)
        # ndv = 255
        # Add Color Table
        # outRaster.GetRasterBand(1).SetRasterColorTable(ct)
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        # Write Date to geotiff
        outband = outRaster.GetRasterBand(1)

        outband.SetNoDataValue(ndv)
        outband.WriteArray(array)
        outRasterSRS = osr.SpatialReference()
        # outRasterSRS.ImportFromEPSG(4326)
        outRaster.SetProjection(projection_wkt)
        # Close Geotiff
        outband.FlushCache()
        del outRaster

    def gdal_merge_bands(self,tif_list,bands_name_list,outf):
        src0 = gdal.Open(tif_list[0])
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(outf,
                               src0.RasterXSize,
                               src0.RasterYSize,
                               len(tif_list),
                               gdal.GDT_Float32)

        out_ds.SetGeoTransform(src0.GetGeoTransform())
        out_ds.SetProjection(src0.GetProjection())
        for idx, tif in enumerate(tif_list, start=1):
            src = gdal.Open(tif)
            band = src.GetRasterBand(1).ReadAsArray()
            out_ds.GetRasterBand(idx).WriteArray(band)
            out_ds.GetRasterBand(idx).SetDescription(bands_name_list[idx - 1])

        out_ds.FlushCache()
        out_ds = None

def main():
    Expand_points_to_rectangle().run()
    Download_from_GEE().run()
    pass


if __name__ == '__main__':
    main()