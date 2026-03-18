sudo -v && \
mojo build -g --debug-info-language=C -O0 ./train_tokenizer.mojo -o tokenizer.out && \
sudo perf record -g -F 99 ./tokenizer.out -v 500 "$@" && \
sudo perf script > out.perf && \
~/FlameGraph/stackcollapse-perf.pl out.perf | \
  LC_ALL=C sed 's/[<＜]/\&lt;/g; s/[>＞]/\&gt;/g' > out.folded && \
~/FlameGraph/flamegraph.pl out.folded > flamegraph.svg && \
firefox flamegraph.svg
