#!/bin/bash

# 1. Check for required filename
if [ -z "$1" ]; then
    echo "Usage: $0 <filename.mojo> [optimization_flag]"
    echo "Example: $0 model.mojo -O3"
    exit 1
fi

TARGET_FILE=$1
# 2. Set default -O0, but allow override from the second argument
OPT_FLAG=${2:-O0}
# Derive binary name (removes .mojo extension)
BIN_NAME="${TARGET_FILE%.*}"

echo "Building $TARGET_FILE with $OPT_FLAG..."

sudo -v && \
mojo build -g --debug-info-language=C -Xlinker -lm "$OPT_FLAG" "$TARGET_FILE" && \
sudo perf record -g -F 99 "./$BIN_NAME" -v 500 "${@:3}" && \
sudo perf script > out.perf && \
~/FlameGraph/stackcollapse-perf.pl out.perf | \
  LC_ALL=C sed 's/[<＜]/\&lt;/g; s/[>＞]/\&gt;/g' > out.folded && \
~/FlameGraph/flamegraph.pl out.folded > flamegraph.svg && \
firefox flamegraph.svg
