import rasterio
import numpy as np
import tables as tab
import tqdm.notebook as tqdm
from skimage.transform import  AffineTransform
from rasterio.transform import Affine
from skimage.measure import ransac
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, SIFT
from PIL import Image
import re
import math, requests
import io
import tqdm
import sys

MATCHING_CUTOFF = 0.7
SEP_CUTOFF = 0.01

def adjust_lat_lon(path_to_PRISMA_file, outpath=""):
    '''
    path_to_PRISMA_file - should be the full path so that re works
    '''
    zoom_level = 11
    tile_width = 7
    tile_height = 7
    
    
    metrics = {}
    print("# 1 create .tif file")
    print("Here")
    tiff_output = outpath + '.tif'
    mean_lat, mean_lon = prisma_2_tiff(path_to_PRISMA_file, tiff_output)
    
    print("# 2 Download gmaps for the same location")
    gmd = GoogleMapDownloader(mean_lat, mean_lon, zoom_level,
                              GoogleMapsLayers.SATELLITE)
    img, tile_coords_corner = gmd.generateImage(tile_width=tile_width,
                                                tile_height=tile_height)
    
    print(" 3 Determine matched features in PRISMA image to gMaps image with SIFT")
    tif_coords, gmaps_coords, nmatches = \
        match_subblocks(tiff_output, tile_coords_corner, img, cutoff=MATCHING_CUTOFF)
    metrics['n_matches'] = nmatches
    del img
    
    orig_errs = np.sqrt(np.sum((tif_coords-gmaps_coords)**2, axis=1))
    metrics['orig_mean_error'] = orig_errs.mean()
    
    keep = orig_errs < SEP_CUTOFF
    src = np.array(tif_coords[keep])
    dst = np.array(gmaps_coords[keep])
    
    
    print(" 4 Define mapping between PRISMA LatLon and gMaps latlon")
    mapping, inliers = ransac((src, dst), AffineTransform, min_samples=4,
                               residual_threshold=0.002, max_trials=100)
    new_errs = np.sqrt(np.sum((mapping(src)-dst)**2, axis=1))
    metrics['new_mean_error'] = new_errs.mean()
    
    n_outliers = np.sum(~inliers)
    metrics['n_outliers'] = n_outliers
    
    print(" 5 Map PRISMA Latlons to gMaps latlons (ransac affine) and save for reference")
    orig = rasterio.open(tiff_output)
    adjusted_latlon = mapping(np.vstack([orig.read(2).flatten(),
                                        orig.read(3).flatten()]).transpose())
    orig.close()
    adjusted_latlon = adjusted_latlon.reshape((1000,1000,2))
    np.savez(outpath + "_adjll.npz", adjusted_latlon = adjusted_latlon)
    return adjusted_latlon, metrics
    
class GoogleMapsLayers:
    ROADMAP = "v"
    TERRAIN = "p"
    ALTERED_ROADMAP = "r"
    SATELLITE = "s"
    TERRAIN_ONLY = "t"
    HYBRID = "y"

def pixel_to_latlon(px_x, px_y, tl_x, tl_y, zoom_level):
    """
        px_x, px_y: Pixel indices of google maps image
        tl_x, tl_y: tile indices of top left (noth-western) corner
        zoom_level: zoom level of the image

        returns: lat,lon
    """
    # background:
    # https://developers.google.com/maps/documentation/javascript/coordinates
    # https://en.wikipedia.org/wiki/Web_Mercator_projection

    tile_size = 256

    world_coord_x = (tile_size*tl_x + px_x) / 2**zoom_level
    world_coord_y = (tile_size*tl_y + px_y) / 2**zoom_level
    
    print(world_coord_x, world_coord_y)

    lon = (2.0*math.pi*world_coord_x / 256.0 - math.pi) * 180.0 / math.pi
    lat = (2.0*math.atan(math.e**math.pi * math.e**(- 2.0*math.pi*world_coord_y / 256.0)) - math.pi/2.0) * 180.0 / math.pi

    return lat,lon

### code for interfacing with google maps
class GoogleMapDownloader:
    """
        A class which generates high resolution google maps images given
        a longitude, latitude and zoom level
    """

    def __init__(self, lat, lng, zoom=12, layer=GoogleMapsLayers.ROADMAP):
        """
            GoogleMapDownloader Constructor
            Args:
                lat:    The latitude of the location required
                lng:    The longitude of the location required
                zoom:   The zoom level of the location required, ranges from 0 - 23
                        defaults to 12
        """
        self._lat = lat
        self._lng = lng
        self._zoom = zoom
        self._layer = layer

    def getXY(self):
        """
            Generates an X,Y tile coordinate based on the latitude, longitude
            and zoom level
            Returns:    An X,Y tile coordinate
        """

        tile_size = 256

        # Use a left shift to get the power of 2
        # i.e. a zoom level of 2 will have 2^2 = 4 tiles
        numTiles = 1 << self._zoom

        # Find the x_point given the longitude
        point_x = (tile_size / 2 + self._lng * tile_size / 360.0) * numTiles // tile_size

        # Convert the latitude to radians and take the sine
        sin_y = math.sin(self._lat * (math.pi / 180.0))

        # Calulate the y coorindate
        point_y = ((tile_size / 2) + 0.5 * math.log((1 + sin_y) / (1 - sin_y)) * -(
        tile_size / (2 * math.pi))) * numTiles // tile_size

        return int(point_x), int(point_y)


    def generateImage(self, **kwargs):
        """
            Generates an image by stitching a number of google map tiles together.
            Args:
                start_x:        The top-left x-tile coordinate
                start_y:        The top-left y-tile coordinate
                tile_width:     The number of tiles wide the image should be -
                                defaults to 5
                tile_height:    The number of tiles high the image should be -
                                defaults to 5
            Returns:
                A high-resolution Goole Map image.
        """

        start_x = kwargs.get('start_x', None)
        start_y = kwargs.get('start_y', None)
        tile_width = kwargs.get('tile_width', 16)
        tile_height = kwargs.get('tile_height', 16)

        # Check that we have x and y tile coordinates
        if start_x == None or start_y == None:
            start_x, start_y = self.getXY()
        # Determine the size of the image
        width, height = 256 * tile_width, 256 * tile_height
        # Create a new image of the size require
        j = 1
        map_img = Image.new('RGB', (width, height))
        for x in range(-tile_width//2, tile_width//2 + 1):
            for y in range(-tile_height//2, tile_height//2 +1):
                url = f'https://mt0.google.com/vt?lyrs={self._layer}&x=' + str(start_x + x) + \
                       '&y=' + str(start_y + y) + '&z=' + str(self._zoom)
                current_tile = str(x) + '-' + str(y)
                response = requests.get(url, stream=True)
                current_tile_pseudofile = io.BytesIO(response.raw.read())
                im = Image.open(current_tile_pseudofile)
                map_img.paste(im, ((x+tile_width//2) * 256, (y+tile_height//2) * 256))
        print('Image size (pix): ', map_img.size)
        tile_coords_corner = start_x-tile_width//2, start_y-tile_height//2
        print(f'Tile coordinate top left (north-west) corner: {tile_coords_corner[0]},{tile_coords_corner[1]}')
        return map_img, (tile_coords_corner)


def pixel_to_latlon(px_x, px_y, tl_x, tl_y, zoom_level):
    """
        px_x, px_y: Pixel indices of google maps image
        tl_x, tl_y: tile indices of top left (noth-western) corner
        zoom_level: zoom level of the image

        returns: lat,lon
    """
    # background:
    # https://developers.google.com/maps/documentation/javascript/coordinates
    # https://en.wikipedia.org/wiki/Web_Mercator_projection

    tile_size = 256

    world_coord_x = (tile_size*tl_x + px_x) / 2**zoom_level
    world_coord_y = (tile_size*tl_y + px_y) / 2**zoom_level
    
    #print(world_coord_x, world_coord_y)

    lon = (2.0*math.pi*world_coord_x / 256.0 - math.pi) * 180.0 / math.pi
    lat = (2.0*math.atan(math.e**math.pi * math.e**(- 2.0*math.pi*world_coord_y / 256.0)) - math.pi/2.0) * 180.0 / math.pi

    return lat,lon

### Function for converting the PRISMA file to a .tiff

def prisma_2_tiff( prisma_path, tiff_output, bandforRGB = 30 ):
    # this function converts a PRISMA image to a geotiff file
    # note that it also fixes a bug in the orientation of the PRISMA image
    file1 = tab.open_file(prisma_path)
    d2 = file1.get_node("/HDFEOS/SWATHS/PRS_L1_HRC/")
    latv = np.array(d2['Geolocation Fields']['Latitude_VNIR'])
    lonv = np.array(d2['Geolocation Fields']['Longitude_VNIR'])
    vnir = d2['Data Fields']['VNIR_Cube'][:,bandforRGB,:]
    file1.close()
    
    lat_centered= latv - latv.mean()
    lat_rescaled = lat_centered / lat_centered.max()
    
    lon_centered= lonv - lonv.mean()
    lon_rescaled = lon_centered / -lon_centered.min()
    
    angle = np.arctan2(lat_rescaled, lon_rescaled, dtype=np.float32)
    
    x = np.linspace(-1, 1, 1000)
    y = np.linspace(-1, 1, 1000)
    coords = np.array(np.meshgrid(x, y), dtype=np.float32).transpose((1, 2, 0))
    
    theta = -angle[500,0] + np.pi/2
    r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    c2 = coords @ r
    
    base_factor = np.array([lat_centered.max(), lon_centered.max()])
    fudge_factor = np.array([c2[:,:,0].max(), -c2[:,:,1].max()]) # this is done because the input lat/lon are transposed
    rescale_factor = base_factor/fudge_factor
    
    means = np.array([latv.mean(), lonv.mean()])
    
    x = np.arange(1000, dtype=np.float32)
    y = np.arange(1000, dtype=np.float32)
    coords = np.array(np.meshgrid(x, y), dtype=np.float32).transpose((1, 2, 0))
    
    to_normalized = Affine.scale(1/500, 1/500)*Affine.translation(-500, -500)
    rot = Affine.rotation(-angle[0,500]*180/np.pi )
    from_normalized = Affine.translation(*(means[::-1]))*Affine.scale(*(rescale_factor[::-1]))#*
    transform = from_normalized*rot*to_normalized
    
    lon, lat = transform * coords.transpose(2, 0, 1)
    
    with rasterio.open(
        tiff_output,
        mode="w",
        driver="GTiff",
        height=lat.shape[0],
        width=lat.shape[1],
        count=3,
        dtype=lat.dtype,
        crs="+proj=latlong",
        transform=transform,
    ) as new_dataset:
        new_dataset.write(vnir, 1)
        new_dataset.write(lat, 2)
        new_dataset.write(lon, 3)
        new_dataset.close()
        
    return latv.mean(), lonv.mean()

### code for matching a PRISMA image tiff file to a gmaps image

def latlon_to_pix(lat, lon, tl_x, tl_y, zoom_level):
    """
        lat, lon: you know
        tl_x, tl_y: tile indices of top left (noth-western) corner
        zoom_level: zoom level of the image

        returns: pixelx pixely
    """
    # background:
    # https://developers.google.com/maps/documentation/javascript/coordinates
    # https://en.wikipedia.org/wiki/Web_Mercator_projection

    tile_size = 256

    world_coord_x = 128*(lon/180+1)
    world_coord_y = (math.pi-np.log(np.tan(math.pi/360*lat+math.pi/4)))*128/(math.pi)
    
    px = 2**zoom_level*world_coord_x - tile_size*tl_x
    py = 2**zoom_level*world_coord_y - tile_size*tl_y
    #print(world_coord_x, world_coord_y)

    #lon = (2.0*math.pi*world_coord_x / 256.0 - math.pi) * 180.0 / math.pi
    #lat = (2.0*math.atan(math.e**math.pi * math.e**(- 2.0*math.pi*world_coord_y / 256.0)) - math.pi/2.0) * 180.0 / math.pi

    return px, py

def match_subblocks(tiff_file, tile_coords_corner, gmap_img, cutoff=MATCHING_CUTOFF):
    blocks = [[[100*i,100*(i+2)-1],[100*j,100*(j+2)-1]] for i in range(9) for j in range(9)]
    
    # get the gmaps image
    gm0 = rgb2gray(gmap_img)
    gm0 = gm0 - gm0.min()
    gm0 /= gm0.max()
    
    # open the tiff file
    orig = rasterio.open(tiff_file)
    pim = np.log(orig.read(1)) - np.log(orig.read(1)).min()
    pim /= pim.max()
    
    nmatches = []
    lla = []
    llb = []
    
    for i in tqdm.tqdm(blocks):
    # determine bounds for PRISMA sub-image
        bounds = [
            [i[0][0],i[1][0]],
            [i[0][1],i[1][0]],
            [i[0][1],i[1][1]],
            [i[0][0],i[1][1]]
        ]

        # determine relevant portion of Gmaps to compare
        def k1_ll(px, py):
            return orig.read(2)[px, py], orig.read(3)[px, py]

        def k2_ll(px, py):
            latlon = pixel_to_latlon(px,py, *tile_coords_corner ,11)
            return latlon

        pbounds = []
        for bound in bounds:
            pbounds.append(latlon_to_pix(*k1_ll(*bound), *tile_coords_corner,11))
        pbounds = np.array(pbounds)
        
        xstart, xend = int(np.floor(pbounds[:,0].min())), int(np.ceil(pbounds[:,0].max()))
        ystart, yend = int(np.floor(pbounds[:,1].min())), int(np.ceil(pbounds[:,1].max()))

        # plot the two maps side by side (and rotated!)
        #fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        #ax[0].imshow(pim[i[0][0]:i[0][1],i[1][0]:i[1][1]])
        #ax[1].imshow(np.rot90(gm0[ystart:yend,xstart:xend]))
        #plt.show()

        # extract descriptors for matching images
        descriptor_extractor = SIFT()
        try: #to deal with runtime errors which can be generated by SIFT
            descriptor_extractor.detect_and_extract(pim[i[0][0]:i[0][1],i[1][0]:i[1][1]])
            keypoints1 = descriptor_extractor.keypoints
            descriptors1 = descriptor_extractor.descriptors
            descriptor_extractor.detect_and_extract((gm0[ystart:yend,xstart:xend]))
            keypoints2 = descriptor_extractor.keypoints
            descriptors2 = descriptor_extractor.descriptors

            matches12 = match_descriptors(descriptors1, descriptors2, max_ratio=cutoff,
                                  cross_check=True)
            
            
            #print("{} matches!".format(len(matches12)))

            # define local distance metrics
            def k2_ll(px, py):
                # not sure why the x and y coordinates switch here
                latlon = pixel_to_latlon(py + xstart ,px + ystart,
                                         *tile_coords_corner ,11)
                return latlon

            def k1_ll(px, py):
                px = px + i[0][0]
                py = py + i[1][0]
                return orig.read(2)[px, py], orig.read(3)[px, py]

            # check to see how close the points are in lat/lon, should usually have .00x accuracy

            for pair in matches12:
                ll1 = k1_ll(*keypoints1[pair[0]])
                lla.append(ll1)
                ll2 = k2_ll(*keypoints2[pair[1]])
                llb.append(ll2)

            nmatches.append(len(matches12))
        except RuntimeError:
            pass
        
    lla = np.array(lla)
    llb = np.array(llb)
    return lla, llb, nmatches    
""" print(sys.argv[1])
latlon, metrics = adjust_lat_lon(sys.argv[1])
print(metrics) """
