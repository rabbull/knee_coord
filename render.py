import trimesh
from bone import BoneCoordination
import numpy as np
from PIL import Image
import pyrender as pyr
import os
import config
import av
import tqdm


def gen_orthographic_photo(meshes: list[trimesh.Trimesh], coord: BoneCoordination, res: tuple[int, int],
                           xmag: float, ymag: float, camera_pose: np.array, light_intensity: float = 3.0,
                           flags=None) -> Image.Image:
    pyr_scene = pyr.Scene()
    for mesh in meshes:
        mesh = mesh.copy()
        mesh.vertices = coord.project(mesh.vertices)
        pyr_mesh = pyr.Mesh.from_trimesh(mesh)
        pyr_scene.add(pyr_mesh)
    pyr_camera = pyr.OrthographicCamera(xmag, ymag, znear=0.1, zfar=1e5)
    pyr_scene.add(pyr_camera, pose=camera_pose)
    pyr_light = pyr.DirectionalLight(color=np.ones(3), intensity=light_intensity)
    pyr_scene.add(pyr_light, pose=camera_pose)
    renderer = pyr.OffscreenRenderer(viewport_width=res[0], viewport_height=res[1])
    render_flags = pyr.renderer.RenderFlags.NONE if flags is None else flags
    color, _ = renderer.render(pyr_scene, render_flags)
    img = Image.fromarray(color)
    return img


def gen_animation(frames: list[Image.Image], name: str, duration: float):
    gif_path = os.path.join(config.OUTPUT_DIRECTORY, f'{name}.gif')
    mp4_path = os.path.join(config.OUTPUT_DIRECTORY, f'{name}.mp4')

    if not frames or len(frames) == 0:
        print('No frames to generate: {}', gif_path)
        return

    # gif
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration / len(frames) * 1000,
        loop=0
    )

    # mp4
    fps = 24
    total_frames = int(fps * duration)
    n_src = len(frames)
    if n_src == total_frames:
        sel = frames
    else:
        sel = []
        step = (n_src - 1) / (total_frames - 1) if total_frames > 1 else 0
        for i in range(total_frames):
            idx = int(round(i * step))
            sel.append(frames[idx])

    with av.open(mp4_path, mode='w') as container:
        stream = container.add_stream('libx264', rate=fps)
        stream.width, stream.height = sel[0].size
        stream.pix_fmt = 'yuv420p'
        stream.options = {
            'crf': '0',
            'preset': 'veryslow',
        }
        for img in tqdm.tqdm(sel):
            if img.mode != 'RGB':
                img = img.convert('RGB')
            frame = av.VideoFrame.from_image(img)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def calc_extents(meshes: dict[config.BoneType, trimesh.Trimesh], coords: dict[config.BoneType, BoneCoordination],
                 padding: int = 5) -> dict[config.BoneType, list[int]]:
    bases = set(meshes.keys()).intersection(coords.keys())
    extents = {}
    for base in bases:
        mesh = meshes[base]
        coord = coords[base]
        extents[base] = calc_extent(mesh, coord, padding)
    return extents


def calc_extent(extended_tibia_mesh: trimesh.Trimesh, tibia_coord: BoneCoordination, padding: int = 5) -> list[int]:
    proj_tm = extended_tibia_mesh.copy()
    proj_tm.vertices = tibia_coord.project(proj_tm.vertices)
    r, t = np.max(proj_tm.vertices[:, :2], axis=0)
    l, b = np.min(proj_tm.vertices[:, :2], axis=0)
    return [l - padding, r + padding, b - padding, t + padding]
