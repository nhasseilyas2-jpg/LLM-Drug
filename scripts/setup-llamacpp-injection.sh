#!/usr/bin/env bash
# Builds the patched llama-server used by the LLM Injection Runtime Lab.
# Usage: scripts/setup-llamacpp-injection.sh [--cuda] [--vulkan] [--metal] [--test] [--dir PATH] [--skip-checkout]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INJ="$ROOT/llamacpp-injection"
DIR="$ROOT/vendor/llama.cpp"
COMMIT="$(tr -d '[:space:]' < "$INJ/patches/LLAMA_CPP_COMMIT")"
PATCH="$INJ/patches/llama.cpp.patch"
EXTRA=()
RUN_TESTS=0
SKIP_CHECKOUT=0
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

while [ $# -gt 0 ]; do
  case "$1" in
    --cuda) EXTRA+=("-DGGML_CUDA=ON") ;;
    --vulkan) EXTRA+=("-DGGML_VULKAN=ON") ;;
    --metal) EXTRA+=("-DGGML_METAL=ON") ;;
    --test) RUN_TESTS=1 ;;
    --skip-checkout) SKIP_CHECKOUT=1 ;;
    --dir) DIR="$2"; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done

command -v git >/dev/null || { echo "git not found" >&2; exit 1; }
command -v cmake >/dev/null || { echo "cmake not found (need 3.21+ and a C++17 compiler)" >&2; exit 1; }

if [ ! -d "$DIR/.git" ]; then
  mkdir -p "$(dirname "$DIR")"
  git clone https://github.com/ggml-org/llama.cpp.git "$DIR"
fi

if git -C "$DIR" apply --check --reverse "$PATCH" 2>/dev/null; then
  echo "Injection patch already applied."
else
  if [ "$SKIP_CHECKOUT" = 0 ]; then
    DIRTY="$(git -C "$DIR" diff --name-only)"
    if [ -n "$DIRTY" ]; then
      # an older version of the injection patch: every modified file carries our marker -> restore and re-apply
      for f in $DIRTY; do
        if ! grep -q "llm-injection" "$DIR/$f"; then
          echo "$DIR has local changes; commit/stash them or use --dir with a clean checkout." >&2; exit 1
        fi
      done
      echo "Removing a previous version of the injection patch"
      # shellcheck disable=SC2086
      git -C "$DIR" checkout -- $DIRTY
    fi
    git -C "$DIR" cat-file -e "$COMMIT^{commit}" 2>/dev/null || git -C "$DIR" fetch origin "$COMMIT"
    git -C "$DIR" -c advice.detachedHead=false checkout "$COMMIT"
  fi
  git -C "$DIR" apply --whitespace=nowarn "$PATCH"
fi

cp "$INJ/src/llm-injection.h" "$INJ/src/llm-injection.cpp" "$INJ/src/llm-injection-graph.cpp" "$DIR/src/"

cmake -S "$DIR" -B "$DIR/build" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON ${EXTRA[@]+"${EXTRA[@]}"}
cmake --build "$DIR/build" --config Release --target llama-server llama-cvector-generator -j "$JOBS"
echo "Built: $DIR/build/bin/llama-server"

if [ "$RUN_TESTS" = 1 ]; then
  cmake -S "$INJ/test" -B "$ROOT/build/engine-tests" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$ROOT/build/engine-tests" --config Release -j "$JOBS"
  "$ROOT/build/engine-tests/test-llm-injection"
fi

echo "Done. Start the lab with: npm start"
