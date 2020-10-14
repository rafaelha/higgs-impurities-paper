GP_FILES = $(wildcard *.gp)
PDF_GP_FILES = $(GP_FILES:%.gp=%.pdf)
SVG_FILES = $(wildcard *.svg)
PDF_SVG_FILES = $(SVG_FILES:%.svg=%.pdf)

all: $(PDF_SVG_FILES) $(PDF_GP_FILES)

$(PDF_SVG_FILES): %.pdf: %.svg Makefile

	# for inkscape < 1.0
	#inkscainkscape --file=$< --export-pdf=$@
	inkscape --export-filename=$@ --export-type=pdf $<

%.pdf: %.gp plot.cfg
	gnuplot $<

	pdflatex $(<:%.gp=%.tex)

	rm -f $(<:%.gp=%-inc.eps)
	rm -f $(<:%.gp=%-inc-eps-converted-to.pdf)
	rm -f $(<:%.gp=%.aux)
	rm -f $(<:%.gp=%.log)
	rm -f $(<:%.gp=%.tex)
	rm -f fit.log
	rm -f texput.log
