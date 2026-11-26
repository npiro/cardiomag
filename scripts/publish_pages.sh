#!/usr/bin/env bash
# Publish the built viewer to GitHub Pages (branch gh-pages, served at https://<owner>.github.io/<repo>/).
#
#   cardiomag-viewer --out viewer/cardiomag_viewer.html     # build first (needs the raw datasets)
#   scripts/publish_pages.sh [viewer/cardiomag_viewer.html]
#
# The gh-pages branch holds only the built page (index.html) and is force-updated on each publish,
# so main stays code-only. Pages is enabled on first use (needs the GitHub CLI, `gh auth login`).
set -euo pipefail
cd "$(dirname "$0")/.."
SRC="${1:-viewer/cardiomag_viewer.html}"
[[ -f "$SRC" ]] || { echo "built viewer not found: $SRC (run cardiomag-viewer first)"; exit 1; }
REMOTE_URL="$(git remote get-url origin)"
REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
SHA="$(git rev-parse --short HEAD)"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cp "$SRC" "$TMP/index.html"
touch "$TMP/.nojekyll"
cat > "$TMP/README.md" <<EOF
Built CardioMag viewer, published from \`$REPO\` @ $SHA by scripts/publish_pages.sh.
Source code and build instructions: https://github.com/$REPO
EOF
(
  cd "$TMP"
  git init -q -b gh-pages
  git add index.html .nojekyll README.md
  git -c user.name="$(cd - >/dev/null; git config user.name)" -c user.email="$(cd - >/dev/null; git config user.email)" \
      commit -q -m "Viewer build from $SHA"
  git push -q --force "$REMOTE_URL" gh-pages:gh-pages
)
# enable Pages on the gh-pages branch (no-op if already enabled)
gh api -X POST "repos/$REPO/pages" -f "source[branch]=gh-pages" -f "source[path]=/" >/dev/null 2>&1 || \
  gh api -X PUT "repos/$REPO/pages" -f "source[branch]=gh-pages" -f "source[path]=/" >/dev/null 2>&1 || true
URL="$(gh api "repos/$REPO/pages" -q .html_url 2>/dev/null || echo "https://${REPO%%/*}.github.io/${REPO##*/}/")"
echo "published $SRC ($(du -h "$SRC" | cut -f1)) -> $URL  (first deploy can take a minute or two)"
