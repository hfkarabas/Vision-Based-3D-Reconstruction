"""Camera checkbox and C shortcut share the same non-destructive overlay state."""
import json
import pytest

from vision3d.demo.viewer import OverlayToggle, CameraControl
from vision3d.demo.support import write_camera_visibility


class Renderer:
    def __init__(self):
        self.cloud = object()
        self.geometries = {self.cloud}
        self.actions = []

    def add_geometry(self, geometry, **kwargs):
        self.geometries.add(geometry)
        self.actions.append(('add', geometry))
        return True

    def remove_geometry(self, geometry, **kwargs):
        self.geometries.remove(geometry)
        self.actions.append(('remove', geometry))
        return True


def test_camera_on_off_on_preserves_cloud_and_independent_axes():
    renderer = Renderer()
    cameras, axes = object(), object()
    active, changes, resets = {}, [], []
    camera = OverlayToggle(cameras, active, lambda vis: resets.append(vis), lambda: changes.append('camera'))
    axis = OverlayToggle(axes, active, lambda vis: None)
    axis.toggle(renderer)
    for desired in [True, False, True]:
        camera.set_visible(renderer, desired)
        assert camera.shown is desired
        assert (cameras in renderer.geometries) is desired
        assert renderer.cloud in renderer.geometries and axes in renderer.geometries
    camera.set_visible(renderer, True)  # Idempotent: no reload, duplicate geometry or reset.
    assert changes == ['camera'] * 3 and len(resets) == 3
    camera.toggle(renderer)
    assert not camera.shown and axes in renderer.geometries


def test_control_file_and_keyboard_remain_synchronized(tmp_path):
    renderer = Renderer()
    toggle = OverlayToggle(object(), {}, lambda vis: None)
    path = tmp_path / 'control.json'
    control = CameraControl(path, toggle)
    def poll():
        control.next_poll = 0
        control.poll(renderer)
    poll()  # No command yet.
    for visible in (True, False, True):
        write_camera_visibility(path, visible)
        poll()
        assert toggle.shown is visible
    old = path.read_text()
    toggle.toggle(renderer)  # C switches off after the GUI's ON command.
    poll()  # Old command must not override the keyboard action.
    assert not toggle.shown
    write_camera_visibility(path, True)
    assert path.read_text() != old  # A new request with the same desired state is still processed.
    poll()
    assert toggle.shown
    assert renderer.cloud in renderer.geometries


@pytest.mark.parametrize('content', ['{', '{}', '{"show_cameras": "yes"}', '[]'])
def test_invalid_commands_do_not_disrupt_viewer(tmp_path, content):
    renderer = Renderer()
    toggle = OverlayToggle(object(), {}, lambda vis: None)
    path = tmp_path / 'control.json'
    path.write_text(content)
    assert CameraControl(path, toggle).poll(renderer) is False
    assert not toggle.shown and renderer.geometries == {renderer.cloud}


def test_visibility_requires_boolean_and_failed_renderer_keeps_state(tmp_path):
    toggle = OverlayToggle(object(), {}, lambda vis: None)
    with pytest.raises(ValueError):
        toggle.set_visible(Renderer(), 1)
    with pytest.raises(ValueError):
        write_camera_visibility(tmp_path / 'control.json', 'yes')
    renderer = Renderer()
    renderer.add_geometry = lambda *args, **kwargs: False
    with pytest.raises(RuntimeError):
        toggle.set_visible(renderer, True)
    assert not toggle.shown
