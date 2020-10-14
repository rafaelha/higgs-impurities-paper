NAME=paper_v1
PREV=

all: $(NAME).pdf

%.pdf: %.tex literature.bib figures/* Makefile
	cd figures && make
	pdflatex $<
	bibtex $(<:%.tex=%.aux)
	pdflatex $<
	pdflatex $<
	make clean


diff: $(NAME).tex literature.bib figures/* Makefile
	latexdiff --type=UNDERLINE $(PREV) $(NAME).tex > $(NAME)_diff.tex
	pdflatex $(NAME)_diff.tex
	bibtex $(NAME)_diff.aux
	pdflatex $(NAME)_diff.tex
	pdflatex $(NAME)_diff.tex
	make clean
	rm -f $(NAME)_diff.tex
	rm -f *.cb
	rm -f *.cb2


submit: $(NAME).tex literature.bib figures/* Makefile
	rm -rf submit/*
	cd figures && make
	pdflatex $(NAME).tex
	bibtex $(NAME).aux
	pdflatex $(NAME).tex
	pdflatex $(NAME).tex
	mkdir -p submit
	cp $(NAME).tex submit/paper.tex
	cp $(NAME).bbl submit/paper.bbl
	cp figures/*.pdf submit/

	# Insert bibliography inline
	sed -i -e '/\\bibliography{literature}/r submit/paper.bbl' submit/paper.tex
	sed -i -e 's/\\bibliography{literature}//g' submit/paper.tex
	rm submit/paper.bbl

	# Remove figure subfolder
	sed -i -e 's/{figures\//{/g' submit/paper.tex

	# Rename figures
	I=0; \
	for X in $$(cat submit/paper.tex | grep -oP '{.*\.pdf}'); do \
		F=$${X%\}}; \
		F=$${F/\{}; \
		F=$${F%.pdf}; \
		F2=figure$$(printf "%02d" $$I); \
		mv submit/$$F.pdf submit/$$F2.pdf; \
		sed -i -e "s/{$$F\.pdf}/{$$F2\.pdf}/g" submit/paper.tex; \
		I=$$(expr $$I + 1); \
	done;

	# Remove unused figures
	for F in $$(ls submit/*.pdf); do \
		if [[ $$(basename $$F) != figure* ]]; then \
			rm $$F; \
		fi; \
	done;

	# Remove comments
	sed -i '/^%/ d' submit/paper.tex

	# Remove double linespaces
	cat -s submit/paper.tex > submit/tmp
	rm submit/paper.tex
	mv submit/tmp submit/paper.tex

	make clean

clean:
	rm -f *.log *.aux *.toc *.bbl *.blg *.out *.acn *.acr *.alg *.glg *.glo *.gls *.glsdefs *.ist *-blx.bib *.run.xml *Notes.bib
