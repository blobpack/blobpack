.PHONY: check test format build publish check-tag

check:
	uv run --group dev ruff check .
	uv run --group dev ruff format --check .

test:
	uv run --group dev --with-editable . pytest -q

format:
	uv run --group dev ruff format .
	uv run --group dev ruff check --fix .

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
