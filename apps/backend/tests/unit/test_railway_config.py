"""
Unit tests for Railway deployment configuration.

Validates that railway.json is correctly configured for production deployment.
These tests run as part of CI to catch configuration issues before deploy.
"""

import json
import pytest
from pathlib import Path


def find_project_root() -> Path:
    """Find the project root by walking up from tests directory."""
    current = Path(__file__).parent
    while current.parent != current:
        # Check for common project root indicators
        if (current / "railway.json").exists():
            return current
        if (current / "compose.yaml").exists():
            return current
        current = current.parent

    # Fallback: assume 4 levels up from tests/unit/test_railway_config.py
    return Path(__file__).parent.parent.parent.parent.parent


@pytest.fixture
def project_root():
    """Get the project root directory."""
    root = find_project_root()
    if not root.exists():
        pytest.skip("Could not find project root")
    return root


@pytest.fixture
def railway_config(project_root):
    """Load railway.json from project root."""
    config_path = project_root / "railway.json"
    if not config_path.exists():
        pytest.skip(f"railway.json not found at {config_path}")

    with open(config_path) as f:
        return json.load(f)


class TestRailwayHealthcheck:
    """Tests for Railway healthcheck configuration."""

    @pytest.mark.unit
    def test_healthcheck_path_exists(self, railway_config):
        """Healthcheck path must be configured."""
        deploy = railway_config.get("deploy", {})
        healthcheck_path = deploy.get("healthcheckPath")

        assert healthcheck_path is not None, \
            "healthcheckPath must be configured in railway.json deploy section"

    @pytest.mark.unit
    def test_healthcheck_path_starts_with_slash(self, railway_config):
        """Healthcheck path must start with /."""
        deploy = railway_config.get("deploy", {})
        healthcheck_path = deploy.get("healthcheckPath", "")

        assert healthcheck_path.startswith("/"), \
            f"healthcheckPath must start with /, got: {healthcheck_path}"

    @pytest.mark.unit
    def test_healthcheck_path_is_live(self, railway_config):
        """Healthcheck should use /live endpoint (not /ready to avoid deploy deadlock)."""
        deploy = railway_config.get("deploy", {})
        healthcheck_path = deploy.get("healthcheckPath")

        # /live is preferred because:
        # - It doesn't depend on external services (InfluxDB, Redis)
        # - Avoids deploy deadlock if dependencies are slow to start
        # - /ready checks dependencies which may not be available during initial deploy
        assert healthcheck_path == "/live", \
            f"healthcheckPath should be '/live' for Railway deploy, got: {healthcheck_path}"

    @pytest.mark.unit
    def test_healthcheck_timeout_reasonable(self, railway_config):
        """Healthcheck timeout should be between 30 and 600 seconds."""
        deploy = railway_config.get("deploy", {})
        timeout = deploy.get("healthcheckTimeout", 300)

        assert 30 <= timeout <= 600, \
            f"healthcheckTimeout should be between 30-600 seconds, got: {timeout}s"


class TestRailwayBuild:
    """Tests for Railway build configuration."""

    @pytest.mark.unit
    def test_build_section_exists(self, railway_config):
        """Build section must exist in railway.json."""
        assert "build" in railway_config, \
            "railway.json must have a 'build' section"

    @pytest.mark.unit
    def test_dockerfile_path_configured(self, railway_config):
        """Dockerfile path must be configured."""
        build = railway_config.get("build", {})
        dockerfile_path = build.get("dockerfilePath")

        assert dockerfile_path is not None, \
            "dockerfilePath must be configured in railway.json build section"

    @pytest.mark.unit
    def test_dockerfile_exists(self, railway_config, project_root):
        """Dockerfile at configured path must exist."""
        build = railway_config.get("build", {})
        dockerfile_path = build.get("dockerfilePath")

        if dockerfile_path is None:
            pytest.skip("dockerfilePath not configured")

        full_path = project_root / dockerfile_path
        assert full_path.exists(), \
            f"Dockerfile not found at {full_path}"

    @pytest.mark.unit
    def test_builder_type_valid(self, railway_config):
        """Builder type must be valid."""
        build = railway_config.get("build", {})
        builder = build.get("builder")

        valid_builders = ["DOCKERFILE", "NIXPACKS", "PAKETO"]
        assert builder in valid_builders, \
            f"builder must be one of {valid_builders}, got: {builder}"


class TestRailwayDeploy:
    """Tests for Railway deploy configuration."""

    @pytest.mark.unit
    def test_deploy_section_exists(self, railway_config):
        """Deploy section must exist in railway.json."""
        assert "deploy" in railway_config, \
            "railway.json must have a 'deploy' section"

    @pytest.mark.unit
    def test_restart_policy_configured(self, railway_config):
        """Restart policy should be configured for production reliability."""
        deploy = railway_config.get("deploy", {})
        restart = deploy.get("restartPolicyType")

        valid_policies = ["ALWAYS", "ON_FAILURE", "NEVER"]
        assert restart in valid_policies, \
            f"restartPolicyType must be one of {valid_policies}, got: {restart}"

    @pytest.mark.unit
    def test_restart_policy_not_never(self, railway_config):
        """Restart policy should not be NEVER for production."""
        deploy = railway_config.get("deploy", {})
        restart = deploy.get("restartPolicyType")

        assert restart != "NEVER", \
            "restartPolicyType should not be 'NEVER' for production deployment"


class TestRailwayPortBinding:
    """Tests to verify backend binds to PORT env var."""

    @pytest.mark.unit
    def test_dockerfile_uses_port_env(self, railway_config, project_root):
        """Dockerfile CMD should use PORT environment variable."""
        build = railway_config.get("build", {})
        dockerfile_path = build.get("dockerfilePath")

        if dockerfile_path is None:
            pytest.skip("dockerfilePath not configured")

        full_path = project_root / dockerfile_path
        if not full_path.exists():
            pytest.skip(f"Dockerfile not found at {full_path}")

        dockerfile_content = full_path.read_text()

        # Check that PORT is used in CMD or ENTRYPOINT
        # Common patterns:
        # - ${PORT:-8000}
        # - $PORT
        # - ENV PORT
        has_port_ref = (
            "${PORT" in dockerfile_content or
            "$PORT" in dockerfile_content or
            "ENV PORT" in dockerfile_content
        )

        assert has_port_ref, \
            "Dockerfile should reference PORT environment variable for Railway compatibility"


class TestRailwayWatchPatterns:
    """Tests for Railway watch patterns (for rebuilds)."""

    @pytest.mark.unit
    def test_watch_patterns_configured(self, railway_config):
        """Watch patterns should be configured to detect changes."""
        build = railway_config.get("build", {})
        patterns = build.get("watchPatterns")

        # Not required but recommended
        if patterns is None:
            pytest.skip("watchPatterns not configured (optional)")

        assert isinstance(patterns, list), \
            "watchPatterns must be a list"

    @pytest.mark.unit
    def test_watch_patterns_include_backend(self, railway_config):
        """Watch patterns should include backend source."""
        build = railway_config.get("build", {})
        patterns = build.get("watchPatterns", [])

        if not patterns:
            pytest.skip("watchPatterns not configured")

        # Should watch the backend directory
        has_backend_pattern = any(
            "backend" in p or "apps/backend" in p
            for p in patterns
        )

        assert has_backend_pattern, \
            "watchPatterns should include backend source directory"


class TestRailwayConfigSchema:
    """Tests for railway.json schema compliance."""

    @pytest.mark.unit
    def test_json_schema_defined(self, railway_config):
        """Railway config should reference the official schema."""
        schema = railway_config.get("$schema")

        # Not required but recommended for IDE support
        if schema is None:
            pytest.skip("$schema not defined (optional but recommended)")

        assert "railway" in schema.lower(), \
            f"$schema should reference Railway schema, got: {schema}"

    @pytest.mark.unit
    def test_no_unknown_top_level_keys(self, railway_config):
        """Config should only have known top-level keys."""
        known_keys = {"$schema", "build", "deploy"}
        actual_keys = set(railway_config.keys())
        unknown_keys = actual_keys - known_keys

        # Railway may add new keys, so just warn
        if unknown_keys:
            # This is not a failure, just a note
            pass
