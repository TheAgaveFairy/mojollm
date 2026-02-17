sudo -v && \
mojo build -g2 --debug-info-language=C -O1 -Xlinker -lm main.mojo && \
sudo perf record -g -F 99 ./main -v 500 "$@" && \
sudo perf script > out.perf && \
~/FlameGraph/stackcollapse-perf.pl out.perf | \
  LC_ALL=C sed 's/[<＜]/\&lt;/g; s/[>＞]/\&gt;/g' > out.folded && \
~/FlameGraph/flamegraph.pl out.folded > flamegraph.svg && \
firefox flamegraph.svg
