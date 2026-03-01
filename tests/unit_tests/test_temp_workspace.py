from mokapot.temp_workspace import TempWorkspace


def test_temp_workspace_cleanup_removes_empty_root(tmp_path):
    ws = TempWorkspace(tmp_path)
    assert ws.path.exists()
    assert (tmp_path / ".mokapot-temp").exists()

    ws.cleanup()

    assert not ws.path.exists()
    assert not (tmp_path / ".mokapot-temp").exists()


def test_temp_workspace_cleanup_keeps_root_with_other_runs(tmp_path):
    ws1 = TempWorkspace(tmp_path)
    ws2 = TempWorkspace(tmp_path)
    temp_root = tmp_path / ".mokapot-temp"
    assert temp_root.exists()

    ws1.cleanup()

    assert not ws1.path.exists()
    assert ws2.path.exists()
    assert temp_root.exists()

    ws2.cleanup()
    assert not temp_root.exists()
