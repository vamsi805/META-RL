import forge_env.server.sandbox.isolation as isolation


def test_forced_subprocess_mode(monkeypatch):
    monkeypatch.setenv("FORGE_TOOL_ISOLATION", "subprocess")
    isolation.docker_daemon_reachable.cache_clear()
    assert isolation.resolve_tool_isolation_mode() is isolation.ToolIsolationMode.SUBPROCESS


def test_forced_docker_mode(monkeypatch):
    monkeypatch.setenv("FORGE_TOOL_ISOLATION", "docker")
    isolation.docker_daemon_reachable.cache_clear()
    assert isolation.resolve_tool_isolation_mode() is isolation.ToolIsolationMode.DOCKER
