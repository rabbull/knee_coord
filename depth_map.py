import os
from collections import defaultdict

import numpy as np
import pandas as pd
import tqdm
import trimesh
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from numpy.linalg import svd
from pyrender import RenderFlags
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

import config
from bone import BoneCoordination
from render import gen_animation, gen_orthographic_photo
from utils import normalize, hex_to_rgba1, get_frame_output_directory, Real

depth_cmap = LinearSegmentedColormap.from_list("depth_map", ['blue', 'green', 'yellow', 'red'])


def gen_depth_map_animation(images: dict[config.BoneType, list[Image.Image]]):
    for direction, frames in images.items():
        gen_animation(frames, name=f'depth_map_{direction.value}', duration=config.DEPTH_MAP_DURATION)


def gen_contact_depth_map_extent(extents: dict[config.BoneType, list[int]]) -> dict[config.BoneType, list[int]]:
    depth_map_extents = {}
    for base, extent in extents.items():
        largest_side = np.max(np.abs(extent))
        extent = (-largest_side, largest_side, -largest_side, largest_side)
        depth_map_extents[base] = extent
    return depth_map_extents


def calc_contact_deepest_points(
        frame_coordinates,
        frame_contact_component_depth_map_origins,
        frame_contact_component_depth_map_depths,
) -> dict[config.BoneType, list[tuple[None | np.ndarray, None | np.ndarray]]]:
    deepest_points = {}
    for base, coords in frame_coordinates.items():
        deepest_points[base] = \
            do_calc_contact_deepest_points(coords, frame_contact_component_depth_map_origins,
                                           frame_contact_component_depth_map_depths)
    print(deepest_points)
    return deepest_points


def do_calc_contact_deepest_points(
        frame_coordinates,
        frame_contact_component_depth_map_origins,
        frame_contact_component_depth_map_depths,
) -> list[tuple[None | np.ndarray, None | np.ndarray]]:
    n = len(frame_contact_component_depth_map_origins)
    frame_deepest_points = []
    for frame_index in range(n):
        coord = frame_coordinates[frame_index]
        deepest = []
        if (frame_contact_component_depth_map_origins and
                frame_contact_component_depth_map_depths):
            for c_origins, c_depth in zip(
                    frame_contact_component_depth_map_origins[frame_index],
                    frame_contact_component_depth_map_depths[frame_index]
            ):
                if len(c_origins) == 0 or len(c_depth) == 0:
                    continue
                idx = np.argmax(c_depth)
                deepest.append((c_origins[idx], c_depth[idx]))

        left, right = (None, -1e9), (None, -1e9)
        for origin, depth in deepest:
            if coord.project(origin)[0] < 0 and left[1] < depth:
                left = (origin, depth)
            if coord.project(origin)[0] > 0 and right[1] < depth:
                right = (origin, depth)

        frame_deepest_points.append((left[0], right[0]))

    return frame_deepest_points


def plot_contact_depth_maps(extents,
                            backgrounds,
                            heatmap_masks,
                            frame_coordinates,
                            frame_bone_distance_map_origins,
                            frame_bone_distance_map_distances,
                            frame_contact_components,
                            frame_contact_component_depth_map_origins,
                            frame_contact_component_depth_map_depths,
                            frame_deepest_points):
    frames = {}
    for base, extent in extents.items():
        background = backgrounds[base]
        heatmap_mask = heatmap_masks[base]
        deepest_points = frame_deepest_points[base]
        coords = frame_coordinates[base]
        frames[base] = do_plot_contact_depth_maps(
            base.value,
            extent,
            background,
            heatmap_mask,
            coords,
            frame_bone_distance_map_origins,
            frame_bone_distance_map_distances,
            frame_contact_components,
            frame_contact_component_depth_map_origins,
            frame_contact_component_depth_map_depths,
            deepest_points,
        )
    return frames


def do_plot_contact_depth_maps(
        name,
        extent,
        background,
        heatmap_mask,
        frame_coordinates,
        frame_bone_distance_map_origins,
        frame_bone_distance_map_distances,
        frame_contact_components,
        frame_contact_component_depth_map_origins,
        frame_contact_component_depth_map_depths,
        frame_deepest_points,
):
    n = len(frame_bone_distance_map_distances)
    res = config.DEPTH_MAP_RESOLUTION
    grid_x, grid_y = np.mgrid[extent[0]:extent[1]:res[0] * 1j, extent[2]:extent[3]:res[1] * 1j]
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
    distance_threshold = config.DEPTH_MAP_DEPTH_THRESHOLD
    vmin, vmax = 1e9, -1e9
    exclude_frames = set()
    for frame_index in range(n):
        distances = frame_bone_distance_map_distances[frame_index]
        distances = distances[(~np.isnan(distances)) & (distances < distance_threshold)]
        depths = frame_contact_component_depth_map_depths
        frame_contact_depths = depths[frame_index] if depths else []

        if len(frame_contact_depths) > 0:
            g_depth = np.concatenate(frame_contact_depths)
            g_depth = g_depth[~np.isnan(g_depth)]
            all_data = np.concatenate([-distances, g_depth])
        else:
            all_data = -distances

        if len(all_data) == 0:
            exclude_frames.add(frame_index)
            continue
        vmax = max(np.max(all_data), vmax)
        vmin = min(np.min(all_data), vmin)
    if all(i in exclude_frames for i in range(n)):
        return None

    frames = []
    for frame_index in tqdm.tqdm(range(n)):
        # if frame_index in exclude_frames:
        #     continue
        coord = frame_coordinates[frame_index]
        origins = frame_bone_distance_map_origins[frame_index].astype(Real)
        distances = frame_bone_distance_map_distances[frame_index]
        mask = (~np.isnan(distances)) & (distances < distance_threshold)
        origins = origins[mask]
        depths = -distances[mask]

        if (frame_contact_components and
                frame_contact_component_depth_map_origins and
                frame_contact_component_depth_map_depths):
            for c_mesh, c_origins, c_depth in zip(
                    frame_contact_components[frame_index],
                    frame_contact_component_depth_map_origins[frame_index],
                    frame_contact_component_depth_map_depths[frame_index]
            ):
                if len(c_origins) == 0 or len(c_depth) == 0:
                    continue
                c_vertices = c_mesh.vertices.astype(Real)
                s_origins = (np.round(origins, decimals=3) * 1e4).astype(np.int64)
                s_vertices = (np.round(c_vertices, decimals=3) * 1e4).astype(np.int64)
                intersect = np.intersect1d(s_origins, s_vertices)
                # o_view = s_origins.view([('', s_origins.dtype)] * 2)
                # v_view = s_vertices.view([('', s_vertices.dtype)] * 2)
                # intersect = np.intersect1d(o_view, v_view)
                keep = np.all(~np.isin(s_origins, intersect), axis=1)
                origins = origins[keep]
                depths = depths[keep]
                origins = np.vstack([origins, c_origins])
                depths = np.concatenate([depths, c_depth])

        origins_projected = coord.project(origins)
        origins_projected_2d = origins_projected[:, :2]

        origins_projected_x = origins_projected[:, 0]
        groups = [origins_projected_x >= 0, origins_projected_x < 0]

        g_origins_2d = [origins_projected_2d[grp_mask] for grp_mask in groups]
        g_depth = [depths[grp_mask] for grp_mask in groups]

        g_z = []
        for i in range(2):
            if (g_origins_2d[i] is not None and len(g_origins_2d[i]) > 1
                    and g_depth[i] is not None and len(g_depth[i]) > 1):
                o, d = g_origins_2d[i], g_depth[i]
                tree = cKDTree(o)
                z = griddata(o, d, (grid_x, grid_y), method='linear')
                dists, _ = tree.query(grid_points)
                z.ravel()[dists > 2] = np.nan
                g_z.append(z)

        if len(g_z) == 2:
            z = np.where(np.isnan(g_z[0]), g_z[1], g_z[0])
            z = np.where(np.isnan(z), g_z[0], z)
        elif len(g_z) == 1:
            z = g_z[0]
        else:
            z = np.full(grid_x.shape, np.nan)

        heatmap_mask_gray = np.array(heatmap_mask.convert("L"))
        z = np.rot90(np.where(heatmap_mask_gray < 128, np.rot90(z), np.nan), k=-1)

        # depth map
        fig, ax = plt.subplots()
        fig.suptitle(f'Depth Map {float(frame_index) / float(n) * 100:.1f}%')
        ax.imshow(background, extent=extent, interpolation='none', aspect='equal')
        im = ax.contourf(
            grid_x, grid_y, z,
            levels=np.arange(vmin, vmax, 1),
            cmap=depth_cmap,
            alpha=0.5,
            extend='both',
        )
        cb = fig.colorbar(im, ax=ax, extend='both')
        cb.set_label('Depth')

        left_deepest_point, right_deepest_point = frame_deepest_points[frame_index]
        if left_deepest_point is not None or right_deepest_point is not None:
            deepest_points = []
            if left_deepest_point is not None: deepest_points.append(left_deepest_point)
            if right_deepest_point is not None: deepest_points.append(right_deepest_point)
            deepest_points = np.array(deepest_points, dtype=Real)
            deepest_points_2d = coord.project(deepest_points)[:, :2]
            ax.scatter(deepest_points_2d[:, 0], deepest_points_2d[:, 1], marker='+', s=100, color='turquoise')

        set_depth_map_axes(ax)

        image_path = os.path.join(get_frame_output_directory(frame_index), f'depth_map_{name}_frame_{frame_index}.jpg')
        directory = os.path.dirname(image_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig.savefig(image_path)
        frames.append(Image.open(image_path))
        plt.close(fig)
    return frames


def get_depth_map_bg_camera_position(base: config.BoneType) -> np.ndarray:
    if base == config.BoneType.TIBIA:
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1000],
            [0, 0, 0, 1],
        ])
    elif base == config.BoneType.FEMUR:
        return np.array([
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, -1000],
            [0, 0, 0, 1],
        ])
    else:
        raise ValueError(f'Unknown bone: {base}')


def depth_map_bg_meshes_generator(bone_color: str, cart_color: str):
    def inner(
            base: config.BoneType,
            bone_meshes: dict[config.BoneType, trimesh.Trimesh],
            cart_meshes: None | dict[config.BoneType, trimesh.Trimesh]) -> list[trimesh.Trimesh]:
        bone_mesh = bone_meshes[base].copy()
        cart_mesh = cart_meshes[base].copy() if cart_meshes is not None and cart_meshes[base] is not None else None
        bone_mesh.visual.vertex_colors = hex_to_rgba1(bone_color)
        meshes = [bone_mesh]
        if cart_mesh is not None:
            cart_mesh.visual.vertex_colors = hex_to_rgba1(cart_color)
            meshes.append(cart_mesh)
        return meshes

    return inner


def gen_contact_depth_map_background(
        extents: dict[config.BoneType, list[int]],
        bone_meshes: dict[config.BoneType, trimesh.Trimesh],
        coords: dict[config.BoneType, BoneCoordination],
        cart_meshes: None | dict[config.BoneType, trimesh.Trimesh] = None) -> dict[config.BoneType, Image.Image]:
    res = config.DEPTH_MAP_RESOLUTION
    intensity = config.DEPTH_MAP_LIGHT_INTENSITY
    backgrounds = {}
    gen_meshes = depth_map_bg_meshes_generator(config.DEPTH_MAP_BONE_COLOR, config.DEPTH_MAP_CARTILAGE_COLOR)
    for base, extent in extents.items():
        coord = coords[base]
        meshes = gen_meshes(base, bone_meshes, cart_meshes)
        pose = get_depth_map_bg_camera_position(base)
        background = gen_orthographic_photo(meshes, coord, res, extent[1], extent[1], pose, intensity)
        if base == config.BoneType.FEMUR:
            background = background.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        backgrounds[base] = background

    return backgrounds


def gen_contact_depth_map_mask(
        extents: dict[config.BoneType, list[int]],
        bone_meshes: dict[config.BoneType, trimesh.Trimesh],
        coords: dict[config.BoneType, BoneCoordination],
        cart_meshes: None | dict[config.BoneType, trimesh.Trimesh] = None) -> dict[config.BoneType, Image.Image]:
    res = config.DEPTH_MAP_RESOLUTION
    masks = {}
    gen_meshes = depth_map_bg_meshes_generator('#FFFFFF', '#000000')
    for base, extent in extents.items():
        coord = coords[base]
        meshes = gen_meshes(base, bone_meshes, cart_meshes)
        pose = get_depth_map_bg_camera_position(base)
        mask = gen_orthographic_photo(meshes, coord, res, extent[1], extent[1], pose, flags=RenderFlags.FLAT)
        if base == config.BoneType.FEMUR:
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        masks[base] = mask

    return masks


def calc_contact_depth_map(frame_ray_directions, frame_contact_components):
    res = []
    for direction, components in tqdm.tqdm(zip(frame_ray_directions, frame_contact_components)):
        if config.DEPTH_BASE_BONE == config.BoneType.FEMUR:
            direction = -direction
        depth_maps = []
        for component in components:
            depth_map = do_calc_contact_depth_map(component, direction)
            depth_maps.append(depth_map)
        res.append(depth_maps)
    return res


def do_calc_contact_depth_map(contact_component, v):
    origins, directions = prepare_rays_from_model(contact_component, v, True)

    # scene = trimesh.Scene([contact_component])
    # for origin, direction in zip(origins, directions):
    #     sphere = icosphere(radius=0.01)
    #     sphere.apply_translation(origin)
    #     scene.add_geometry(sphere)
    #
    #     ud = direction / np.linalg.norm(direction)
    #     end = origin + ud * 0.1
    #     segment = np.stack([origin, end], axis=0)
    #     arrow = cylinder(radius=0.005, segment=segment)
    #     scene.add_geometry(arrow)
    # scene.show()

    locations, ray_indices, _ = \
        contact_component.ray.intersects_location(origins, directions, multiple_hits=False)
    if len(ray_indices) == 0:
        return np.zeros((0, 3)), np.zeros((0,))
    origins = origins[ray_indices]
    depths = np.linalg.norm(locations - origins, axis=1)
    return origins, depths


def calc_bone_distance_map(fcs, tcs, fs, ts, vs):
    if fcs is None or tcs is None or any(fc is None for fc in fcs) or any(tc is None for tc in tcs):
        fcs, tcs = fs, ts  # use bone instead
    if config.DEPTH_BASE_BONE == config.BoneType.FEMUR:
        return [do_calc_bone_distance_map(t, f, -v) for f, t, v in tqdm.tqdm(zip(fcs, tcs, vs))]
    elif config.DEPTH_BASE_BONE == config.BoneType.TIBIA:
        return [do_calc_bone_distance_map(f, t, v) for f, t, v in tqdm.tqdm(zip(fcs, tcs, vs))]
    raise NotImplementedError(f'Unknown base bone: {config.DEPTH_BASE_BONE}')


def do_calc_bone_distance_map(target, base, v):
    origins, directions = prepare_rays_from_model(base, v)
    locations, ray_indices, _ = \
        target.ray.intersects_location(origins, directions, multiple_hits=False)
    origins = origins[ray_indices]
    depths = np.linalg.norm(locations - origins, axis=1)
    return origins, depths


def prepare_rays_from_model(model, direction, inward: bool = False, eps: float = 1e-4):
    ud = normalize(direction)
    mask = np.dot(model.vertex_normals, ud) > 0
    origins = model.vertices[mask]

    if inward:
        ud = -ud
    origins += ud * eps
    directions = np.tile(ud, (len(origins), 1))
    return origins, directions


def plot_max_depth_curve(frame_contact_components, contact_component_depth_maps, frame_coordinates):
    res = {}
    for base, coords in frame_coordinates.items():
        res[base] = do_plot_max_depth_curve(base.value, frame_contact_components, contact_component_depth_maps, coords)
    return res


def do_plot_max_depth_curve(name, frame_contact_components, contact_component_depth_maps, frame_coordinates):
    if not (frame_contact_components and contact_component_depth_maps):
        return None
    n = len(frame_coordinates)
    mdms = []
    mdls = []
    mds = []
    for i in range(n):
        mdm, mdl = 0, 0
        coordinate = frame_coordinates[i]
        components = frame_contact_components[i]
        depth_maps = contact_component_depth_maps[i]
        for c, (_, depths) in zip(components, depth_maps):
            if depths is None or depths.shape[0] == 0:
                continue
            if ((cx := coordinate.project(c.centroid)[0]) >= 0 and config.KNEE_SIDE == config.KneeSide.LEFT) or \
                    (cx < 0 and config.KNEE_SIDE == config.KneeSide.RIGHT):
                mdm = max(mdm, np.max(depths))
            else:
                mdl = max(mdl, np.max(depths))
        mdms.append(mdm)
        mdls.append(mdl)
        mds.append(max(mdm, mdl))

    fig, ax = plt.subplots()
    ax.plot(np.arange(n), mds)
    fig.savefig(os.path.join(config.OUTPUT_DIRECTORY, f'max_depth_curve_{name}.png'))
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(np.arange(n), mdms, label='Medial')
    ax.plot(np.arange(n), mdls, label='Lateral')
    ax.legend()
    fig.savefig(os.path.join(config.OUTPUT_DIRECTORY, f'max_depth_curve_{name}_split.png'))
    plt.close(fig)

    df = pd.DataFrame({
        'index': np.arange(n),
        'max_depth': np.array(mds, dtype=Real),
        'max_depth_medial': np.array(mdms, dtype=Real),
        'max_depth_lateral': np.array(mdls, dtype=Real),
    })
    df.to_csv(os.path.join(config.OUTPUT_DIRECTORY, f'max_depth_curve_{name}.csv'), index=False)

    return mds, mdms, mdls


def plot_min_distance_curve(frame_bone_distance_origins, frame_bone_distances, frame_coordinates):
    res = {}
    for base, coords in frame_coordinates.items():
        res[base] = do_plot_min_distance_curve(base.value, frame_bone_distance_origins, frame_bone_distances, coords)
    return res


def do_plot_min_distance_curve(name, frame_bone_distance_origins, frame_bone_distances, frame_coordinates):
    if frame_bone_distance_origins is None or frame_bone_distances is None or frame_coordinates is None:
        return None
    n = len(frame_coordinates)
    mdms = []
    mdls = []
    mds = []
    for i in range(n):
        mdm, mdl = 0, 0
        coordination = frame_coordinates[i]
        origins = frame_bone_distance_origins[i]
        distances = frame_bone_distances[i]

        origins_x = coordination.project(origins)[:, 0]
        distances_left = distances[origins_x < 0]
        distances_right = distances[origins_x > 0]
        min_distance_left = np.min(distances_left) if len(distances_left) > 0 else 0
        min_distance_right = np.min(distances_right) if len(distances_right) > 0 else 0
        if config.KNEE_SIDE == config.KneeSide.LEFT:
            mdm, mdl = min_distance_right, min_distance_left
        else:
            mdm, mdl = min_distance_left, min_distance_right
        mdms.append(mdm)
        mdls.append(mdl)
        mds.append(min(mdm, mdl))

    fig, ax = plt.subplots()
    ax.plot(np.arange(n), mds)
    fig.savefig(os.path.join(config.OUTPUT_DIRECTORY, f'min_distance_curve_{name}.png'))
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(np.arange(n), mdms, label='Medial')
    ax.plot(np.arange(n), mdls, label='Lateral')
    ax.legend()
    fig.savefig(os.path.join(config.OUTPUT_DIRECTORY, f'min_distance_curve_{name}_split.png'))
    plt.close(fig)

    df = pd.DataFrame({
        'index': np.arange(n),
        'min_distance': np.array(mds, dtype=Real),
        'min_distance_medial': np.array(mdms, dtype=Real),
        'min_distance_lateral': np.array(mdls, dtype=Real),
    })
    df.to_csv(os.path.join(config.OUTPUT_DIRECTORY, f'min_distance_curve_{name}.csv'), index=False)

    return mds, mdms, mdls


def calc_frame_contact_plane_normal_vectors(frame_contact_areas: list[trimesh.Trimesh],
                                            coords: list[BoneCoordination]) -> list[np.ndarray]:
    normal_vectors = []
    for idx, contact_area in enumerate(frame_contact_areas):
        uz = normalize(coords[idx].t.unit_z)
        if contact_area.is_empty:
            normal_vectors.append(uz)
            continue
        vertices = contact_area.vertices
        centroid = contact_area.centroid
        adj_vertices = vertices - centroid
        u, s, vh = svd(adj_vertices)
        normal = normalize(vh[-1, :])
        if normal.dot(uz) > normal.dot(-uz):
            normal = -normal
        normal_vectors.append(normal)
    return normal_vectors


def plot_deepest_points(
        backgrounds: dict[config.BoneType, Image.Image],
        extends: dict[config.BoneType, tuple[float, float, float, float]],
        deepest_points: dict[config.BoneType, list[tuple[None | np.ndarray, None | np.ndarray]]],
        coordinates: dict[config.BoneType, list[BoneCoordination]]
) -> dict[config.BoneType, tuple[np.ndarray, np.ndarray]]:
    res = {}
    for base in backgrounds.keys():
        background = backgrounds[base]
        extent = extends[base]
        frame_deepest_points = deepest_points[base]
        frame_coordinates = coordinates[base]
        if all(p is None for p in frame_deepest_points):
            continue
        res[base] = do_plot_deepest_points(base, background, extent, frame_deepest_points, frame_coordinates)
    return res


def do_plot_deepest_points(
        base: config.BoneType,
        background: Image.Image,
        extent: tuple[float, float, float, float],
        frame_deepest_points: list[tuple[None | np.ndarray, None | np.ndarray]],
        frame_coordinates: list[BoneCoordination],
) -> dict[str, np.ndarray]:
    left_points = []
    right_points = []
    for coord, (left, right) in zip(frame_coordinates, frame_deepest_points):
        if left is not None:
            left = coord.project(left)[:2]
            left_points.append(left)
        if right is not None:
            right = coord.project(right)[:2]
            right_points.append(right)

    if config.KNEE_SIDE == config.KneeSide.LEFT:
        points = {
            'Medial': right_points,
            'Lateral': left_points,
        }
    elif config.KNEE_SIDE == config.KneeSide.RIGHT:
        points = {
            'Medial': left_points,
            'Lateral': right_points,
        }
    else:
        raise ValueError('unreachable')

    fig, ax = plt.subplots()
    ax.imshow(background, extent=extent, interpolation='none', aspect='equal')

    np_pts = {}
    for name, pts in points.items():
        if len(pts) == 0:
            continue
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], marker='+', label=name)
        np_pts[name] = pts
    ax.legend()
    set_depth_map_axes(ax)

    img_path = os.path.join(
        config.OUTPUT_DIRECTORY,
        f'deepest_points_{base.value}.png'
    )
    fig.savefig(img_path)
    plt.close(fig)

    return np_pts


def plot_fixed_points(
        backgrounds: dict[config.BoneType, Image.Image],
        extends: dict[config.BoneType, tuple[float, float, float, float]],
        fixed_points: dict[config.BoneType, dict[str, list[list[float]]]],
        coordinates: dict[config.BoneType, list[BoneCoordination]],
) -> dict[config.BoneType, dict[str, np.ndarray]]:
    res = {}
    for base in backgrounds.keys():
        background = backgrounds[base]
        extent = extends[base]
        frame_fixed_points = fixed_points[base.invert()]
        frame_coordinates = coordinates[base]
        res[base] = do_plot_fixed_points(base, background, extent, frame_fixed_points, frame_coordinates)
    return res


def do_plot_fixed_points(
        base: config.BoneType,
        background: Image.Image,
        extent: tuple[float, float, float, float],
        frame_fixed_points: dict[str, list[list[float]]],
        frame_coordinates: list[BoneCoordination],
) -> dict[str, np.ndarray]:
    points_2d = {}
    for name, points in frame_fixed_points.items():
        points_2d[name] = []
        for coord, point in zip(frame_coordinates, points):
            points_2d[name].append(coord.project(point)[:2])
        points_2d[name] = np.array(points_2d[name])

    fig, ax = plt.subplots()
    ax.imshow(background, extent=extent, interpolation='none', aspect='equal')
    for name, points in points_2d.items():
        ax.plot(points[:, 0], points[:, 1], marker='+', label=name)
    set_depth_map_axes(ax)
    fig.savefig(os.path.join(
        config.OUTPUT_DIRECTORY,
        f'fixed_points_{base}.png'
    ))
    plt.close(fig)

    return points_2d


def set_depth_map_axes(ax):
    for label in ax.get_yticklabels():
        if label.get_text() == '0':
            label.set_text('')
    for label in ax.get_xticklabels():
        if label.get_text() == '0':
            x, y = label.get_position()
            label.set_position((x + 1, y))
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
