"""Exercise camera-path loading and auto-discovery without starting Open3D."""

import numpy as np
import pytest

from src.reconstruction import CameraPath, find_camera_path, load_camera_path


def _rotation(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]], dtype=np.float64)


def _poses(count: int) -> np.ndarray:
    poses = np.tile(np.eye(4), (count, 1, 1))
    for index in range(count):
        poses[index, :3, :3] = _rotation(index * 0.1)
        poses[index, :3, 3] = [index * 0.2, index * -0.1, index * 0.05]
    return poses


def test_saved_centers_and_rotations_reconstruct_the_poses(tmp_path):
    poses = _poses(3)
    source = tmp_path / "point_cloud.npz"
    np.savez(
        source,
        camera_centers=poses[:, :3, 3],
        camera_rotations=poses[:, :3, :3],
        camera_translations=np.linalg.inv(poses)[:, :3],
    )

    path = load_camera_path(source)
    np.testing.assert_allclose(path.camera_to_world, poses, atol=1e-6)
    np.testing.assert_allclose(
        path.world_to_camera @ poses, np.tile(np.eye(4), (len(poses), 1, 1)), atol=1e-6
    )
    np.testing.assert_allclose(path.centers, poses[:, :3, 3])


def test_camera_to_world_predictions_and_directory_discovery(tmp_path):
    poses = _poses(2)
    np.savez(tmp_path / "predictions.npz", camera_to_world=poses)
    path = load_camera_path(tmp_path)
    np.testing.assert_allclose(path.camera_to_world, poses)


def test_world_to_camera_is_inverted_and_intrinsics_are_preserved(tmp_path):
    poses = _poses(2)
    intrinsics = np.stack([np.eye(3) * 500, np.eye(3) * 500])
    source = tmp_path / "cameras.npz"
    np.savez(
        source,
        world_to_camera=np.linalg.inv(poses)[:, :3],
        intrinsics=intrinsics,
    )

    path = load_camera_path(source)
    np.testing.assert_allclose(path.camera_to_world, poses, atol=1e-6)
    np.testing.assert_allclose(path.intrinsics, intrinsics)


def test_auto_discovery_prefers_cameras_then_falls_back_to_predictions(tmp_path):
    cloud = tmp_path / "point_cloud.ply"
    cloud.write_bytes(b"placeholder")
    np.savez(tmp_path / "point_cloud.npz", points=np.zeros((1, 3)))

    with pytest.raises(FileNotFoundError):
        find_camera_path(cloud)

    np.savez(tmp_path / "predictions.npz", camera_to_world=_poses(2))
    assert find_camera_path(cloud).name == "predictions.npz"

    np.savez(
        tmp_path / "cameras.npz",
        centers=_poses(3)[:, :3, 3],
        rotations=_poses(3)[:, :3, :3],
    )
    discovered = load_camera_path(find_camera_path(cloud))
    assert len(discovered) == 3


def test_invalid_camera_pose_shapes_and_rotations_are_rejected(tmp_path):
    source = tmp_path / "camera_to_world.npz"
    np.savez(source, camera_to_world=np.zeros((2, 4, 4)))
    with pytest.raises(ValueError, match="homogeneous"):
        load_camera_path(source)

    bad_rotation = _poses(1)
    bad_rotation[0, :3, :3] *= 2
    with pytest.raises(ValueError, match="orthonormal"):
        CameraPath(bad_rotation)


def test_follow_camera_cli_validation_before_viewer_launch(tmp_path):
    from view_point_cloud import main

    cloud = tmp_path / "point_cloud.ply"
    cloud.write_bytes(b"placeholder")
    with pytest.raises(SystemExit) as error:
        main([str(cloud), "--follow-cameras", str(tmp_path / "missing.npz")])
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        main([str(cloud), "--camera-fps", "0"])
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        main([str(cloud), "--camera-fov", "180"])
    assert error.value.code == 2
