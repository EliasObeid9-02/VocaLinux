#import "../helpers.typ": in_outline, ar, ar_std

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
).rev()

#show outline.entry: it => link(
  it.element.location(),
  ar_std([
    #heading_format(it.element)
    #it.inner()
    #h(1.5em * (it.level - 1))\
  ])
)

#show outline.entry.where(level: 1): it => link(
  it.element.location(),
  ar_std([
    #it.inner()\
  ])
)

#outline(
  title: ar([الفهرس])
)
#pagebreak()

#show outline.entry: it => link(
  it.element.location(),
  it.indented(
    ar_std(it.prefix()),
    ar_std(it.inner())
  )
)

#outline(
  title: ar([فهرس الأشكال]),
  target: figure.where(kind: image)
)
#pagebreak()

#outline(
  title: ar([فهرس الجداول]),
  target: figure.where(kind: table)
)
#pagebreak()
