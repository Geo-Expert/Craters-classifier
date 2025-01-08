import pyproj
import pandas as pd
import random
import rasterio
from functions import *


if __name__ == '__main__':
    base_dir = "C:/Users/dannyp1801/OneDrive - Technion/Second_degree/Courses/lior's course/moon_craters"
    map_file = base_dir + "/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif"
    craters_csv = base_dir + "/lunar_crater_database_robbins_2018_bundle/data/lunar_crater_database_robbins_2018.csv"
    output_dir = "D:/craters_dataset"

    min_diameter = 3
    max_diameter = 10
    latitude_bounds = (-60, 60)
    offset = 0.5

    dst_height, dst_width = 100, 100

    craters_to_output = -1

    # Load the Robbins dataset
    craters = pd.read_csv(craters_csv)
    # Filter craters by diameter and latitude range
    filtered_craters = craters[
        (craters['DIAM_CIRC_IMG'] >= min_diameter) &
        (craters['DIAM_CIRC_IMG'] <= max_diameter) &
        (craters['LAT_CIRC_IMG'] >= latitude_bounds[0]) &
        (craters['LAT_CIRC_IMG'] <= latitude_bounds[1])
        ]

    if craters_to_output > 0:
        random_craters_idx = random.sample(range(len(filtered_craters)), k=craters_to_output)
        filtered_craters = filtered_craters.iloc[random_craters_idx]

    craters_wkt = """
            GEOGCS["GCS_Moon",
            DATUM["D_Moon_2000",
            SPHEROID["Moon_2000_IAU_IAG",1737151.3,0, LENGTHUNIT["metre",1]]],
            PRIMEM["Reference_Meridian",0],
            UNIT["metre",1]],
            PROJECTION["Equirectangular"],
            PARAMETER["standard_parallel_1",0],
            PARAMETER["central_meridian",0],
            PARAMETER["false_easting",0],
            PARAMETER["false_northing",0],
            UNIT["metre",1,
                AUTHORITY["EPSG","9001"]],
            AXIS["Easting",EAST],
            AXIS["Northing",NORTH],
            AUTHORITY["ESRI","103881"]]
            """

    # Create the Moon CRS
    craters_crs = pyproj.CRS.from_wkt(craters_wkt)
    # Create a transformer for lat/lon to the map's coordinate reference system

    with rasterio.open(map_file) as map_ref:

        transformer = pyproj.Transformer.from_crs(craters_crs, map_ref.crs.to_string(), always_xy=True)

        counter = 0
        # Process each crater
        for _, crater in filtered_craters.iterrows():
            counter += 1
            if counter % 100 == 0:
                print(f'processed {counter} craters')
            crop_and_save_crater(
                map_ref,
                crater['CRATER_ID'],
                crater['LAT_CIRC_IMG'],
                crater['LON_CIRC_IMG'],
                crater['DIAM_CIRC_IMG'],
                offset,
                output_dir,
                transformer,
                dst_height,
                dst_width
            )

