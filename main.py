from matplotlib import pyplot
from mpl_toolkits import mplot3d
import numpy as np
import stl

import io
import gzip


def load_stl_gz(filename) -> stl.Mesh:
    compressed_stl_file = open(filename, "rb")
    compressed_stl = compressed_stl_file.read()
    compressed_stl_file.close()
    stl_data = gzip.decompress(compressed_stl)
    mesh = stl.Mesh.from_file('fuck numpy-stl', fh=io.BytesIO(stl_data))

    return mesh


def main():
    femur_stl_filename = "test_case/SUBN_02_Femur_RE_Surface.stl.gz"
    tibia_stl_filename = "test_case/SUBN_02_Tibia_RE_Surface.stl.gz"

    femur_mesh = load_stl_gz(femur_stl_filename)
    tibia_mesh = load_stl_gz(tibia_stl_filename)

    figure = pyplot.figure()
    axes = figure.add_subplot(projection='3d')

    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(femur_mesh.vectors))
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(tibia_mesh.vectors))

    scale = np.concat([femur_mesh.points.flatten(), tibia_mesh.points.flatten()])
    axes.auto_scale_xyz(scale, scale, scale)

    pyplot.show()


if __name__ == '__main__':
    main()
