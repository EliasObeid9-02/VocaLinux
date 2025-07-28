.PHONY: build-en build-ar

build-doc-en:
	typst compile doc/main.typ english_documentation.pdf --input lang="en"

build-doc-ar:
	typst compile doc/main.typ arabic_documentation.pdf --input lang="ar"
