load 'plot.cfg'
set output "example.tex"

set multiplot layout 2,2 \
    margins 0.1,0.9,0.1,0.9 \
    spacing 0.1,0.15

set xrange[0:2*pi]
set yrange[-1.5:2]

set key horizontal
plot sin(x+0.4) ls 1 title 'f1', \
	sin(x+0.8) ls 2 title 'f2', \
	sin(x+1.2) ls 3 title 'f3', \
	sin(x+1.6) ls 4 title 'f4'


set view map
splot sin(5*x)*cos(5*y) title '' w pm3d


plot sin(x+0.4) ls 1 title 'f1', \
	sin(x+0.8) ls 11 title 'f2', \
	sin(x+1.2) ls 21 title 'f3', \
	sin(x+1.6) ls 31 title 'f4'

unset view
splot sin(5*x)*cos(5*y) title '' w pm3d
