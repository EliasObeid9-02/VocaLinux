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
    [#heading_format(it.element) #it.inner() #linebreak()]
  )
)

#show outline.entry.where(level: 1): it => link(
  it.element.location(),
  it.inner()
)

#align(center, outline(title: [Index]))
#pagebreak()

#show outline.entry: it => link(
  it.element.location(),
  it.indented(it.prefix(), it.inner())
)

#align(center, outline(title: [Figure Index#v(0.5em)], target: figure.where(kind: image)))
#pagebreak()

#align(center, outline(title: [Table Index#v(0.5em)], target: figure.where(kind: table)))
#pagebreak()
