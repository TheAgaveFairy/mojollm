mojo build -g --debug-info-language=C -O0 tokenizer.mojo && \
sudo perf record -g -F 99 ./tokenizer && \
sudo perf script > out.perf && \
~/FlameGraph/stackcollapse-perf.pl out.perf > out.folded && \
~/FlameGraph/flamegraph.pl out.folded > flamegraph.svg && \
firefox flamegraph.svg
