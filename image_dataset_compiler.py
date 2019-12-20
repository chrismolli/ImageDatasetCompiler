"""
    :author: Christian Mollière
    :version: 0.1
    :date: 20-12-2019
"""
import pickle
import sys
import urllib.request
from io import BytesIO, TextIOWrapper

import cv2
import numpy as np
import progressbar
import timeout_decorator
from google_images_download import google_images_download

IMAGE_DOWNLOAD_TIMEOUT_SECONDS = 10

class ImageDatasetCompiler:
    """
        Simple class to fetch images from Google,
        crop and rescale them and finally compile a dataset of unlabeled image data.
    """
    _images = []
    _urls = []

    def __init__(self):
        pass

    def _search_image_urls(self, samples_per_keyword, keywords):
        """ searches image urls from google """
        print("Collecting image urls...")
        # empty urls
        self._urls = []
        # switch std
        old_stdout = sys.stdout
        sys.stdout = TextIOWrapper(BytesIO(), sys.stdout.encoding)
        # create response object
        response = google_images_download.googleimagesdownload()
        # prepare query
        arguments = {
            "keywords": keywords,
            "limit": samples_per_keyword,
            "print_urls": True,
            "format": "jpg",
            "no_download": True
        }
        # retrieve paths
        paths = response.download(arguments)
        # get output
        sys.stdout.seek(0)
        output = sys.stdout.read()
        # switch back to old std
        sys.stdout.close()
        sys.stdout = old_stdout
        # reformat output
        for line in output.split("\n"):
            if line.startswith("Image URL:"):
                self._urls.append(line.replace("Image URL: ", ""))
        print("{} (of requested {}) urls found for keywords [{}]".format(len(self._urls),
                                                                         samples_per_keyword * len(keywords.split(',')),
                                                                         keywords))

    @timeout_decorator.timeout(IMAGE_DOWNLOAD_TIMEOUT_SECONDS)
    def _fetch_from_url(self,url):
        """ fetches a single image from an url and saves it to the image list """
        with urllib.request.urlopen(url) as req:
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            return cv2.imdecode(image, cv2.IMREAD_COLOR)

    def _get_images(self, width, height, upscaling):
        """ fetches image data from url """
        print("Fetching {} images...".format(len(self._urls)))
        # empty images
        self._images = []
        # get images
        with progressbar.ProgressBar(max_value=len(self._urls)) as bar:
            for idx, url in enumerate(self._urls):
                try:
                    image = self._fetch_from_url(url)
                    image = self._crop(image, width, height)
                    if image.shape[1] >= width or (image.shape[1] < width and upscaling):
                        # omit to small pictures if upscaling is False
                        image = self._scale(image,width, height)
                        if len(image.shape) == 3 and image.shape[2] == 3:
                            self._images.append(image)
                    bar.update(idx)
                except:
                    print("ImageDatasetCompiler Warning: Could not fetch: {}".format(url))
        print("{} images downloaded!".format(len(self._images)))

    def _crop(self, image, width, height):
        """ crops an image to a wanted aspect ratio """
        wanted_aspect = width / height
        aspect = image.shape[1] / image.shape[0]
        if aspect > wanted_aspect:
            # image to wide -> crop width
            new_width = int(image.shape[0] * wanted_aspect)
            crop = int((image.shape[1] - new_width) / 2)
            return image[:, crop:new_width + crop, :]
        elif aspect < wanted_aspect:
            # image to tall -> crop height
            new_height = int(image.shape[1] / wanted_aspect)
            crop = int((image.shape[0] - new_height) / 2)
            return image[crop:new_height + crop, :, :]
        else:
            return image

    def _scale(self, image, width, height):
        """ scales an image to a wanted size """
        if image.shape[1] != width or image.shape[0] != height:
            return cv2.resize(image, dsize=(height, width), interpolation=cv2.INTER_CUBIC)
        else:
            return image

    def _stack(self):
        """ stacks list of images to final dataset """
        self._images = np.stack(self._images, axis=0)

    def compile(self, samples_per_keyword, keywords, width=100, height=100, up_scaling=True):
        """
        Fetches pictures from Google Images and compiles a image data set for Machine Learning.
        :param samples_per_keyword: Integer, number of pictures to sample from Google, max 100!
        :param query: String, picture label to search for
        :param width: Integer, wanted image width
        :param height: Integer, wanted image height
        :param up_scaling: Boolean, If True to small or to big images will be cut and scaled to fit.
                                    If False to small pictures will be skipped.
        :return x: Numpy Array of Integer, RGB Image data
        """
        print("ImageDatasetCompiler started compiling dataset!")
        samples_per_keyword = min(samples_per_keyword, 100)
        self._search_image_urls(samples_per_keyword, keywords)
        self._get_images(width, height, up_scaling)
        self._stack()
        print("ImageDatasetCompiler finished compiling {} / {} image samples!".format(self._images.shape[0],
                                                                                      samples_per_keyword * len(
                                                                                          keywords.split(','))))
        return self._images

    def compile_from_txt(self, fname, n_samples_per_keyword=100, width=100, height=100, up_scaling=True, seperator=','):
        """ reads keywords from txt file """
        with open(fname, 'r') as file:
            if seperator == ',':
                keywords = file.read()
            else:
                keywords = file.read().replace(seperator, ",")
        self.compile(n_samples_per_keyword, keywords, width, height, up_scaling)
        return self._images

    def save(self, fname):
        """ saves image data to binary file """
        if type(self._images) is np.ndarray:
            with open(fname, 'wb') as file:
                pickle.dump(self._images, file)
                print("ImageDatasetCompiler saved dataset of {} samples to {}".format(self._images.shape[0], fname))
        else:
            print("ImageDatasetCompiler Warning: No compiled dataset found to save. No data written!")

    def load(self, fname):
        """ loads binary dataset to numpy array """
        with open(fname, 'rb') as file:
            self._images = pickle.load(file)
        self._urls = []
        return self._images

    def rescale(self, width, height):
        """ rescales dataset """
        if type(self._images) is np.ndarray:
            print("Rescaling dataset of {} images from ({},{}) to ({},{})".format(self._images.shape[0],
                                                                                  self._images.shape[2],
                                                                                  self._images.shape[1],
                                                                                  width, height))
            rescaled_images = []
            for i in range(self._images.shape[0]):
                image = self._images[i]
                rescaled_images.append(cv2.resize(image, dsize=(height, width), interpolation=cv2.INTER_CUBIC))
            self._images = rescaled_images
            self._stack()
        else:
            print("ImageDatasetCompiler Warning: No valid dataset to rescale available!")

        return self._images