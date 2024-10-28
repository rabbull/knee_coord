import copy
import io
import gzip

import scipy.spatial
import trimesh
import numpy as np
from tqdm import tqdm
import shapely
from trimesh.path import Path2D

trimesh.util.attach_to_log()


def load_stl_gz(filename) -> trimesh.Trimesh:
    with open(filename, "rb") as f:
        compressed_stl = f.read()
    stl_data = gzip.decompress(compressed_stl)
    return trimesh.load(file_obj=io.BytesIO(stl_data), file_type='stl')


def path2d_to_polygon(path_2d: trimesh.path.Path2D) -> shapely.geometry.Polygon:
    exterior_coords = []

    # 遍历 Path2D 的 entities，提取线段并获取端点
    for entity in path_2d.entities:
        # 如果实体是线段类型（Line），获取端点
        if isinstance(entity, trimesh.path.entities.Line):
            for index in entity.points:
                exterior_coords.append(path_2d.vertices[index])

    # 确保路径闭合：第一个点和最后一个点相同
    if len(exterior_coords) > 0 and not (exterior_coords[0] == exterior_coords[-1]).all():
        exterior_coords.append(exterior_coords[0])

    # 构造 Shapely 的 Polygon 对象
    polygon = shapely.geometry.Polygon(exterior_coords)

    return polygon


def fill_hole_advanced(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    boundary = mesh.outline()
    boundary.remove_unreferenced_vertices()
    boundary_2d: trimesh.path.Path2D
    boundary_2d, to_3d = boundary.to_planar()

    polygon = path2d_to_polygon(boundary_2d)
    x_min, y_min, x_max, y_max = polygon.bounds
    step = 0.5
    x_grid = np.arange(x_min, x_max, step)
    y_grid = np.arange(y_min, y_max, step)
    xv, yv = np.meshgrid(x_grid, y_grid)
    raw_grid = np.vstack((xv.flatten(), yv.flatten())).T

    shapely.prepare(polygon)
    mask = shapely.contains_xy(polygon, raw_grid)
    new_vertices_2d = raw_grid[mask]

    all_vertices_2d = np.concatenate([boundary_2d.vertices, new_vertices_2d])
    tri = scipy.spatial.Delaunay(all_vertices_2d)
    new_faces = tri.simplices

    count = boundary_2d.vertices.shape[0]
    a = np.column_stack([boundary_2d.vertices, np.zeros(count), np.ones(count)])
    a = np.dot(to_3d, a.T).T[:, :3]
    print(a[:10])
    print()

    recovered = boundary_2d.to_3D(to_3d)
    print(recovered.vertices[:10])
    print()

    count = all_vertices_2d.shape[0]
    all_vertices_2d_homo = np.column_stack([all_vertices_2d, np.zeros(count), np.ones(count)])
    new_vertices = np.dot(to_3d, all_vertices_2d_homo.T).T[:, :3]
    print(new_vertices[:10])


def fill_hole_simple(orig_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    num_original_vertices = orig_mesh.vertices.shape[0]
    print(f'{num_original_vertices} vertices are contained in original mesh.')
    print(f'{orig_mesh.faces.shape[0]} faces are contained in original mesh.')

    boundary_3d = orig_mesh.outline()
    boundary_3d.remove_unreferenced_vertices()
    boundary_2d = Path2D(
        entities=copy.deepcopy(boundary_3d.entities),
        vertices=copy.deepcopy(boundary_3d.vertices)[:, :2],
    )
    z = copy.deepcopy(boundary_3d.vertices[0, 2])

    boundary_polygon = path2d_to_polygon(boundary_2d)
    x_min, y_min, x_max, y_max = boundary_polygon.bounds
    step = 0.5
    x_grid = np.arange(x_min, x_max, step)
    y_grid = np.arange(y_min, y_max, step)
    xv, yv = np.meshgrid(x_grid, y_grid)
    raw_grid = np.vstack((xv.flatten(), yv.flatten())).T
    shapely.prepare(boundary_polygon)
    mask = shapely.contains_xy(boundary_polygon, raw_grid)
    new_vertices_2d = raw_grid[mask]
    print(f'{new_vertices_2d.shape[0]} new points are generated.')
    new_vertices_z = z * np.ones((new_vertices_2d.shape[0], 1))

    new_faces_vertices = np.concatenate([boundary_2d.vertices, new_vertices_2d])
    new_faces = scipy.spatial.Delaunay(new_faces_vertices).simplices
    print(f'{len(new_faces)} new faces are generated.')

    new_faces_index_mapping = {}
    for i, j in enumerate(range(boundary_2d.vertices.shape[0], new_faces_vertices.shape[0])):
        new_faces_index_mapping[j] = num_original_vertices + i
    for i in orig_mesh.outline().entities[0].points:
        original_vertex = orig_mesh.vertices[i]
        for j, vertex_2d in enumerate(boundary_2d.vertices):
            if np.linalg.norm(vertex_2d - original_vertex[:2]) < 1e-6:
                new_faces_index_mapping[j] = i
    assert len(new_faces_index_mapping) == new_faces_vertices.shape[0]

    new_vertices_3d = np.concatenate([new_vertices_2d, new_vertices_z], axis=1)
    new_mesh_vertices = np.concatenate([orig_mesh.vertices, new_vertices_3d])
    print(f'{new_mesh_vertices.shape[0]} vertices are contained in new mesh.')
    new_mesh_faces = np.concatenate([orig_mesh.faces, np.zeros((len(new_faces), 3), dtype=np.int64)])
    for index, face in enumerate(new_faces):
        i, j, k = face[0], face[1], face[2]
        new_mesh_faces[index + orig_mesh.faces.shape[0]] = (
            new_faces_index_mapping[i],
            new_faces_index_mapping[j],
            new_faces_index_mapping[k],
        )
    print(f'{new_mesh_faces.shape[0]} faces are contained in new mesh.')

    return trimesh.Trimesh(
        vertices=new_mesh_vertices,
        faces=new_mesh_faces,
    )


def fill_hole(mesh: trimesh.Trimesh):
    outline = mesh.outline()
    line = outline.entities[0]
    points: np.ndarray = line.points
    n = len(points)
    assert n > 2
    prev, i, j = 0, 1, n - 1
    new_faces = []
    while i != j:
        new_faces.append([points[prev], points[i], points[j]])
        prev, i = i, i + 1
        if i == j:
            break
        new_faces.append([points[prev], points[i], points[j]])
        prev, j = j, j - 1
        if i == j:
            break
    assert len(new_faces) == n - 2
    new_faces = np.array(new_faces, dtype='int64')
    mesh.faces = np.concatenate([mesh.faces, new_faces])

    return mesh


def fill_hole_afm(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    from CGAL import CGAL_Advancing_front_surface_reconstruction as af
    from CGAL.CGAL_Kernel import Point_3
    from CGAL.CGAL_Point_set_3 import Point_set_3
    from CGAL.CGAL_Polyhedron_3 import Polyhedron_3

    vertices = Point_set_3()
    polyhedron = Polyhedron_3()

    for x, y, z in tqdm(mesh.vertices):
        vertices.insert(Point_3(float(x), float(y), float(z)))

    af.advancing_front_surface_reconstruction(vertices, polyhedron)

    new_vertices = np.zeros(shape=(polyhedron.size_of_vertices(), 3), dtype='float64')
    vertex_indices = {}
    for index, vertex in enumerate(tqdm(polyhedron.vertices(), total=polyhedron.size_of_vertices())):
        new_vertices[index] = np.array([vertex.x(), vertex.y(), vertex.z()]).astype('float64')
        vertex_indices[vertex] = index

    new_faces = []
    for facet in tqdm(polyhedron.facets(), total=polyhedron.size_of_facets()):
        halfedge = facet.halfedge()
        indices = []
        h = halfedge
        while True:
            vertex = h.vertex()
            idx = vertex_indices[vertex]
            indices.append(idx)
            h = h.next()
            if h == halfedge:
                break
        # 处理三角形和非三角形面片
        if len(indices) == 3:
            new_faces.append(indices)
        else:
            for i in range(1, len(indices) - 1):
                new_faces.append([indices[0], indices[i], indices[i + 1]])
    new_faces = np.array(new_faces, dtype='int64')

    new_mesh = trimesh.Trimesh(new_vertices, new_faces)
    return new_mesh


def fill_hole_polygon(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    path = mesh.outline()
    path.remove_unreferenced_vertices()
    path_2d = Path2D(
        entities=copy.deepcopy(path.entities),
        vertices=copy.deepcopy(path.vertices)[:, :2],
    )
    z = copy.deepcopy(path.vertices[0, 2])
    cap = trimesh.creation.extrude_polygon(path2d_to_polygon(path_2d), height=z)
    return mesh + cap


def main2():
    femur_stl_filename = "test_case/SUBN_02_Femur_RE_Surface.stl.gz"
    tibia_stl_filename = "test_case/SUBN_02_Tibia_RE_Surface.stl.gz"

    femur_mesh = load_stl_gz(femur_stl_filename)
    femur_mesh.export('femur_orig.stl')
    tibia_mesh = load_stl_gz(tibia_stl_filename)
    tibia_mesh.export('tibia_orig.stl')

    # femur_mesh = fill_hole_simple(femur_mesh)
    # femur_mesh.show()
    # print(femur_mesh.is_watertight)

    # scene = trimesh.Scene([
    #     femur_mesh,
    #     tibia_mesh
    # ])
    # scene.show()


def main():
    femur_stl_filename = 'ContactPara/Assic/Femur_Cartilage.stl'
    tibia_lateral_stl_filename = 'ContactPara/Assic/Tibia_Cartilage_Lateral.stl'
    tibia_lateral_stl_filename = 'ContactPara/TibialCartilagr.stl'
    tibia_medial_stl_filename = 'ContactPara/Assic/Tibia_Cartilage_Medial.stl'
    femur_mesh = trimesh.load_mesh(femur_stl_filename)
    tibia_lateral_mesh = trimesh.load_mesh(tibia_lateral_stl_filename)
    tibia_medial_mesh = trimesh.load_mesh(tibia_medial_stl_filename)

    for mesh in [femur_mesh, tibia_lateral_mesh, tibia_medial_mesh]:
        mesh: trimesh.Trimesh
        trimesh.repair.fill_holes(mesh)
        print(mesh.outline())
        print(mesh, mesh.is_watertight)
    return

    intersection_lateral = trimesh.boolean.intersection([tibia_lateral_mesh, femur_mesh])
    intersection_medial = trimesh.boolean.intersection([tibia_medial_mesh, femur_mesh])

    scene = trimesh.Scene([intersection_medial, intersection_lateral])
    scene.show()
    return

    z = np.array([0, 0, 1], dtype=np.float64)
    scene = trimesh.Scene()
    for part in intersection.split():
        scene.add_geometry(part)
        vertex_mask = part.vertex_normals[:, 2] < 0
        filtered_vertices = part.vertices[vertex_mask]
        origins = filtered_vertices
        # directions = np.array(-z, dtype=np.float64).repeat(len(origins)).reshape(-1, 3)
        directions = -part.vertex_normals[vertex_mask]
        locations, ray_indices, _ = part.ray.intersects_location(origins, directions)
        origins = origins[ray_indices]
        distances = np.linalg.norm(origins - locations, axis=1)
        vertex_index = ray_indices[np.argmax(distances)]
        vertex = part.vertices[vertex_index]

        visual = trimesh.creation.uv_sphere(radius=0.3)
        visual.apply_translation(vertex)
        scene.add_geometry(visual)
    scene.show()


if __name__ == '__main__':
    main()
