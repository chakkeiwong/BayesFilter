# Saved first-case diagnostic only; this script performs no simulation.
# Run from the repository root with gnuplot.
set terminal pdfcairo enhanced color size 7.0in,3.25in font "Sans,10"
set output "docs/figures/neutra_controlled_valley_residuals.pdf"
set xlabel "Physical first coordinate x_1"
set ylabel "Transformed score-residual norm"
set xrange [-2:2]
set logscale y
set format y "10^{%L}"
set grid ytics lc rgb "#dddddd"
set key outside right center title "Second coordinate"
set border 3
set tics nomirror
set style line 1 lc rgb "#0072B2" lw 1.6
set style line 2 lc rgb "#56B4E9" lw 1.6
set style line 3 lc rgb "#222222" lw 2.1
set style line 4 lc rgb "#E69F00" lw 1.6
set style line 5 lc rgb "#D55E00" lw 1.6
plot for [column=2:6] "docs/figures/neutra_controlled_valley_residuals.dat" \
     using 1:column with lines ls (column-1) title sprintf("x_2 = %d",column-4)
