"""Python-launched Viser viewer; full SH rendering runs on the development server."""

import argparse
import math
import threading
import time
from pathlib import Path

import numpy as np
import torch

from ..rendering.cli import load_cameras, rendering_settings
from ..rendering.renderer import render_gaussians, require_cuda
from .contracts import Camera
from .storage import load_model


def view_splat(
    model_path: str | Path,
    *,
    device: str = "cuda",
    host: str = "127.0.0.1",
    port: int = 8080,
    max_image_side: int = 1280,
    fps: float = 15,
    cameras_path: str | Path | None = None,
):
    import viser
    from viser import transforms as tf

    require_cuda(device)
    if max_image_side < 16 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Viewer resolution must be >= 16 and fps must be finite and positive")
    model, metadata = load_model(model_path, device)
    config, degree = rendering_settings(metadata)
    cameras = load_cameras(cameras_path) if cameras_path else []
    if not cameras and "initial_camera" in metadata:
        cameras = [Camera.from_dict(metadata["initial_camera"])]
    if not cameras:
        raise ValueError("PLY has no initial camera; supply --cameras with at least one view")

    server = viser.ViserServer(host=host, port=port)
    server.scene.world_axes.visible = False
    initial = cameras[0].camera_to_world
    # VGGT's world need not have a gravity axis. Use the first capture's up.
    server.scene.set_up_direction(tuple(-initial[:3, 1]))
    lock = threading.Lock()
    pending: dict[int, tuple] = {}
    clients: dict[int, object] = {}

    def queue_camera(client):
        camera = client.camera
        if camera.update_timestamp == 0 or camera.aspect <= 0 or camera.fov <= 0:
            return
        # Snapshot only camera state in callbacks. GPU work happens in one thread;
        # newer snapshots replace old ones instead of building an unbounded queue.
        with lock:
            if client.client_id in clients:
                pending[client.client_id] = (
                    camera.wxyz.copy(),
                    camera.position.copy(),
                    float(camera.fov),
                    float(camera.aspect),
                )

    def set_capture_camera(client, camera):
        pose = camera.camera_to_world
        with client.atomic():
            client.camera.position = pose[:3, 3]
            client.camera.look_at = pose[:3, 3] + pose[:3, 2]
            client.camera.up_direction = -pose[:3, 1]
            client.camera.wxyz = tf.SO3.from_matrix(pose[:3, :3].copy()).wxyz
            client.camera.fov = 2 * math.atan(camera.height / (2 * camera.intrinsics[1, 1]))
        queue_camera(client)

    @server.on_client_connect
    def on_connect(client):
        with lock:
            clients[client.client_id] = client
        client.gui.add_markdown(
            "### Gaussian splat\nOrbit: drag · Pan: right-drag · Zoom: scroll\n\n"
            "RGB is rendered on the server with the stored spherical harmonics. "
            "Capture presets set pose/FOV; the viewport uses a centred pinhole camera."
        )
        options = [f"{i}: {camera.frame_id}" for i, camera in enumerate(cameras)]
        selected = client.gui.add_dropdown("Capture camera", options=options)

        @selected.on_update
        def on_preset(_):
            set_capture_camera(client, cameras[options.index(selected.value)])

        @client.camera.on_update
        def on_camera(_):
            queue_camera(client)

        # Browser camera state is populated by its first websocket camera message.
        # Initialize exactly once after that state is available.
        initialized = False

        @client.camera.on_update
        def on_first_camera(_):
            nonlocal initialized
            if not initialized:
                initialized = True
                set_capture_camera(client, cameras[0])

        if client.camera.update_timestamp:
            on_first_camera(None)

    @server.on_client_disconnect
    def on_disconnect(client):
        with lock:
            clients.pop(client.client_id, None)
            pending.pop(client.client_id, None)

    print(f"Viewer listening at http://{host}:{port}; Ctrl+C to stop", flush=True)
    try:
        while True:
            started = time.monotonic()
            with lock:
                requests = list(pending.items())
                pending.clear()
            for client_id, (wxyz, position, fov, aspect) in requests:
                with lock:
                    client = clients.get(client_id)
                if client is None:
                    continue
                if aspect >= 1:
                    width, height = max_image_side, max(1, round(max_image_side / aspect))
                else:
                    width, height = max(1, round(max_image_side * aspect)), max_image_side
                focal = height / (2 * math.tan(fov / 2))
                k = np.array([[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]])
                camera_to_world = np.eye(4)
                camera_to_world[:3, :3] = tf.SO3(wxyz).as_matrix()
                camera_to_world[:3, 3] = position
                camera = Camera(str(client_id), width, height, k, np.linalg.inv(camera_to_world))
                with torch.no_grad():
                    rendered = render_gaussians(model, camera, config, sh_degree=degree)
                    image = rendered.rgb.clamp(0, 1).mul(255).round().byte().cpu().numpy()
                with lock:
                    connected = client_id in clients
                if connected:
                    client.scene.set_background_image(image, format="jpeg")
            time.sleep(max(0.001, 1 / fps - (time.monotonic() - started)))
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--max-image-side", default=1280, type=int)
    parser.add_argument("--fps", default=15.0, type=float)
    parser.add_argument("--cameras", type=Path)
    args = parser.parse_args()
    view_splat(
        args.model,
        device=args.device,
        host=args.host,
        port=args.port,
        max_image_side=args.max_image_side,
        fps=args.fps,
        cameras_path=args.cameras,
    )
