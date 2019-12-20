"""
    Example for the use of ImageDatasetCompiler
"""

"""
    Create a new compiler instance
"""
from image_dataset_compiler import ImageDatasetCompiler
compiler = ImageDatasetCompiler()

""" 
    Start compiler to search, download, crop and scale images.
    Use up_scaling=False to omit images that are smaller than the wanted size
"""
compiler.compile(10,"snow, trees, mountains",width=32,height=32)

"""
    Save dataset to binary file using pickle.
"""
compiler.save("dataset.bin")
