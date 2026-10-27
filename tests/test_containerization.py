"""Tests for Phase 2 Milestone P1: Containerization."""

from pathlib import Path

import yaml


def test_dockerignore_structure_and_safety() -> None:
    """Verify .dockerignore exists and excludes secrets and heavy local directories."""
    dockerignore_path = Path(".dockerignore")
    assert dockerignore_path.is_file(), ".dockerignore must exist"

    content = dockerignore_path.read_text(encoding="utf-8")
    lines = {
        line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")
    }

    # Critical security and confidentiality exclusions
    assert ".env" in lines or ".env*" in lines
    assert "PLAN.md" in lines
    assert "STORY.md" in lines
    assert ".git" in lines
    assert ".venv" in lines
    assert "__pycache__" in lines


def test_dockerfile_multi_stage_and_security() -> None:
    """Verify Dockerfile contains multi-stage build, non-root user, and healthcheck."""
    dockerfile_path = Path("Dockerfile")
    assert dockerfile_path.is_file(), "Dockerfile must exist"

    content = dockerfile_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines()]

    # Multi-stage check: at least two FROM statements
    from_statements = [line for line in lines if line.startswith("FROM")]
    assert len(from_statements) >= 2, "Dockerfile must use a multi-stage build"

    # Non-root user check
    has_user = any("USER appuser" in line or "USER 1000" in line for line in lines)
    assert has_user, "Dockerfile must specify a non-root USER for runtime security"

    # Healthcheck check
    has_healthcheck = any(line.startswith("HEALTHCHECK") for line in lines)
    assert has_healthcheck, "Dockerfile must define a container HEALTHCHECK"

    # Default build target argument
    has_build_arg = any("BUILD_TARGET" in line for line in lines)
    assert has_build_arg, "Dockerfile must support BUILD_TARGET build arg (e.g. cpu/gpu)"


def test_docker_compose_services_and_healthchecks() -> None:
    """Verify docker-compose.yml defines api, qdrant, postgres, and redis with healthchecks."""
    compose_path = Path("docker-compose.yml")
    assert compose_path.is_file(), "docker-compose.yml must exist"

    content = compose_path.read_text(encoding="utf-8")
    data = yaml.safe_load(content)

    assert "services" in data, "docker-compose.yml must define services"
    services = data["services"]

    # Required Phase 2 services
    assert "api" in services
    assert "qdrant" in services
    assert "postgres" in services
    assert "redis" in services

    # Verify healthchecks are configured for backing services
    assert "healthcheck" in services["qdrant"], "Qdrant must have a healthcheck"
    assert "healthcheck" in services["postgres"], "PostgreSQL must have a healthcheck"
    assert "healthcheck" in services["redis"], "Redis must have a healthcheck"

    # Verify api depends on healthy infrastructure services
    api_depends = services["api"].get("depends_on", {})
    assert "qdrant" in api_depends, "API service must depend on qdrant"

    # Verify persistent volumes are declared
    volumes = data.get("volumes", {})
    assert "qdrant_storage" in volumes
    assert "postgres_data" in volumes
    assert "redis_data" in volumes
