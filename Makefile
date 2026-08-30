.PHONY: check test format perf build publish check-tag

check:
	uv run --group dev ruff check .
	uv run --group dev ruff format --check .

test:
	uv run --group dev --with-editable . pytest -q

format:
	uv run --group dev ruff format .
	uv run --group dev ruff check --fix .

# pre-release workload gate (wall-clock, so not a required CI check);
# run before pushing a release tag
perf:
	uv run --group dev --with-editable . python benchmarks/perf_gate.py

build:
	rm -rf dist
	uv build

# hatch-vcs derives the package version from the tag itself, so tag and
# version cannot disagree; this only refuses tags that are not plain
# vX.Y.Z releases (extend the pattern if a pre-release is ever wanted)
check-tag:
	echo "$(TAG)" | grep -Eq '^v[0-9]+\.[0-9]+\.[0-9]+$$' || { echo "not a release tag: $(TAG)"; exit 1; }

# release.yml runs this from a tag with an OIDC token; no secret involved
publish:
	uv publish --trusted-publishing always
