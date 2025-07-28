#import "../helpers.typ": in_outline

#set outline.entry(fill: line(length: 100%))
#show outline: it => {
  in_outline.update(true)
  it
  in_outline.update(false)
}

#let heading_format(element) = context str(
  counter(heading).
  at(element.location()).
  map(str).join("-")
)

#show outline.entry: it => link(
  it.element.location(),
  it.indented(
    none,
    [#heading_format(it.element) #it.inner()]
  )
)

#show outline.entry.where(level: 1): it => link(
  it.element.location(),
  it.indented(
    none,
    it.inner()
  )
)

#outline(
  title: [Index#v(0.5em)],
  indent: 2em
)
#pagebreak()

#show outline.entry: it => link(
  it.element.location(),
  it.indented(it.prefix(), it.inner())
)

#outline(
  title: [Figure Index#v(0.5em)],
  indent: 2em,
  target: figure.where(kind: image)
)
#pagebreak()

#outline(
  title: [Table Index#v(0.5em)],
  indent: 2em,
  target: figure.where(kind: table)
)
#pagebreak()
