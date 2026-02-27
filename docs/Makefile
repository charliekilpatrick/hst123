# Minimal Sphinx Makefile; output goes to build/html for GitHub Pages publish_dir.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = hst123
SOURCEDIR     = .
BUILDDIR      = build

.PHONY: help Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
