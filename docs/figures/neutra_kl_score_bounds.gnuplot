# Deterministic bounds derived in ch26f; no fitted or stochastic data.
set terminal pdfcairo enhanced color font "DejaVu Sans,10" size 6.6in,2.7in
set output "neutra_kl_score_bounds.pdf"
set logscale x 10
set logscale y 10
set xrange [10:1e6]
set samples 500
set format x "10^{%L}"
set format y "10^{%L}"
set border 3
set xtics nomirror
set ytics nomirror
set grid back lc rgb "#dddddd"
set multiplot layout 1,2 margins .10,.97,.23,.90 spacing .12,.04
set xlabel "Oscillation frequency k"
set ylabel "KL upper bound (nats)"
set key top right font ",8"
plot 1/x title "KL(p_k || Gaussian)" lw 2 lc rgb "#2864a0", \
     1/(2*x*(1-1/sqrt(x))**2) title "KL(Gaussian || p_k)" lw 2 dt 2 lc rgb "#bd613b"
set ylabel "Score-residual RMS lower bound"
unset key
plot sqrt(x/2)/(1+1/sqrt(x)) lw 2 lc rgb "#9d3045"
unset multiplot
