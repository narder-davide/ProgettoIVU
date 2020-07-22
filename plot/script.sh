#!/usr/bin/gnuplot -persist
set terminal epslatex color lw 2
set output 'image.tex'
#set ylabel "HR"
set xlabel "Epoch" 
set title "Accuracy"
#set lmargin 0
#set title "Class $1$ stationary distribution"
set size 0.8,0.8
set xrange [1:200]
#set xtics  rotate by -45
set key bottom right
plot "best.log" using 1:2 title "Train" w lines lc "black",\
"best.log" using 1:3 title "Test" w lines lc "red" dt 3
#    EOF

 
