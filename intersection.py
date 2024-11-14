import trimesh

from utils import *

trimesh.util.attach_to_log()


def main():
    femur_stl_filename = 'ContactPara/Femur_Cartilage_fixed_3.stl'
    tibia_lateral_stl_filename = 'ContactPara/Assic/Tibia_Cartilage_Lateral.stl'
    tibia_medial_stl_filename = 'ContactPara/Assic/Tibia_Cartilage_Medial.stl'
    femur_mesh: trimesh.Trimesh = trimesh.load_mesh(femur_stl_filename)
    tibia_lateral_mesh: trimesh.Trimesh = trimesh.load_mesh(tibia_lateral_stl_filename)
    tibia_medial_mesh: trimesh.Trimesh = trimesh.load_mesh(tibia_medial_stl_filename)

    # scene = trimesh.Scene([femur_mesh, tibia_lateral_mesh, tibia_medial_mesh])
    # scene.show()

    femur_mesh = extend_mesh(femur_mesh, True)
    femur_mesh.show()
    # tibia_lateral_mesh = extend_mesh(tibia_lateral_mesh, -Z)
    # tibia_medial_mesh = extend_mesh(tibia_medial_mesh, -Z)

    scene = trimesh.Scene([femur_mesh, tibia_lateral_mesh, tibia_medial_mesh])
    scene.show()
    return

    intersection_lateral = trimesh.boolean.intersection([tibia_lateral_mesh, femur_mesh])
    intersection_medial = trimesh.boolean.intersection([tibia_medial_mesh, femur_mesh])

    scene = trimesh.Scene([intersection_medial, intersection_lateral])
    scene.show()
    return

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
