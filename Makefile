.PHONY: build-en build-ar

build-doc-en:
	typst compile doc/main.typ Transcribing-Speech-into-Linux-Terminal-Commands.pdf --input lang="en"

build-doc-ar:
	typst compile doc/main.typ Transcribing-Speech-into-Linux-Terminal-Commands.pdf --input lang="ar"
