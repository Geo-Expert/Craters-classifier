import rasterio
from rasterio.windows import from_bounds, transform
import pyproj
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import rasterio
import cv2
from numpy import cos, radians


def rotate_crater_by_shadow_angle(crater_image):
    pass


def crop_and_save_crater(map_ref, crater_id, lat, lon, diameter, offset, output_dir, transformer, dst_h, dst_w):

    if lon > 180:
        lon -= 360
    # Convert latitude and longitude to map's coordinate system
    x, y = transformer.transform(lon, lat)
    # Define bounding box in projected coordinates
    radius = (diameter / 2) * 1000  # Convert km to meters
    radius_with_offset_x = (radius + radius * offset) / cos(radians(lat))
    radius_with_offset_y = radius + radius * offset
    min_x, min_y = x - radius_with_offset_x, y - radius_with_offset_y
    max_x, max_y = x + radius_with_offset_x, y + radius_with_offset_y

    # Get the window for cropping
    window = from_bounds(min_x, min_y, max_x, max_y, transform=map_ref.transform)

    # Read and crop the data
    cropped_image = map_ref.read(window=window)

    cropped_image = cropped_image.reshape((cropped_image.shape[1], cropped_image.shape[2]))

    projected_height = int(cropped_image.shape[0] / cos(radians(abs(lat))))
    if projected_height > cropped_image.shape[0]:
        cropped_image_projected = cv2.resize(cropped_image, (cropped_image.shape[1], projected_height))
    else:
        cropped_image_projected = cropped_image

    resized_image = cv2.resize(cropped_image_projected, (dst_w, dst_h))

    flipped_image = flip_crater(resized_image)


    plt.subplot(1, 4, 1)
    plt.imshow(cropped_image, cmap='gray')
    plt.title(f'cylindrical')
    plt.subplot(1, 4, 2)
    plt.imshow(cropped_image_projected, cmap='gray')
    plt.title(f'conformal')
    plt.subplot(1, 4, 3)
    plt.imshow(resized_image, cmap='gray')
    plt.title(f'resized')
    plt.subplot(1, 4, 4)
    plt.imshow(flipped_image, cmap='gray')
    plt.title(f'shadow flipped')
    plt.suptitle(f'diamitter:{round(diameter, 0)}, lat:{round(lat, 0)}')
    plt.show()

    # filename = f"{output_dir}/{crater_id}.jpeg"
    # plt.imsave(filename, resized_image, cmap='gray')


def flip_crater(img):
    '''
    Flips crater s.t. the shadow will always be on the r.h.s
    '''
    qtr_img_width = np.int16(img.shape[1] / 4)
    half_img_width = np.int16(img.shape[1] / 2)

    left_crater_side = img[:, qtr_img_width:half_img_width]
    right_crater_side = img[:, half_img_width:-qtr_img_width]

    if left_crater_side.mean() > right_crater_side.mean():
        pass
    else:
        img = np.fliplr(img)

    return img


def reflect_crater(img):
    qtr_img_width = np.int16(img.shape[1] / 4)
    half_img_width = np.int16(img.shape[1] / 2)

    # Find side with shadow
    left_side = img[:, :half_img_width]
    right_side = img[:, half_img_width:]

    left_crater_side = img[:, qtr_img_width:half_img_width]
    right_crater_side = img[:, half_img_width:-qtr_img_width]

    if left_crater_side.mean() > right_crater_side.mean():
        img[:, half_img_width:] = np.fliplr(left_side)
    else:
        img[:, :half_img_width] = np.fliplr(right_side)

    return img


if __name__ == '__main__':
    base_dir = "/Users/danny/Library/CloudStorage/OneDrive-Technion/Second_degree/Courses/lior's course/moon_craters"
    map_file = base_dir + "/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif"
    craters_csv = base_dir + "/lunar_crater_database_robbins_2018_bundle/data/lunar_crater_database_robbins_2018.csv"
    output_dir = base_dir + "/craters_dataset"

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

