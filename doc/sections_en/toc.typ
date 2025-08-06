#import "../helpers.typ": in_outline, en, en_std

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
  en_std([
    #h(1.5em * (it.level - 1))
    #heading_format(it.element)
    #it.inner()\
  ])
)

#show outline.entry.where(level: 1): it => link(
  it.element.location(),
  en_std([
    #it.inner()\
  ])
)

#outline(
  title: en([Index])
)
#pagebreak()

#show outline.entry: it => link(
  it.element.location(),
  it.indented(
    en_std(it.prefix()),
    en_std(it.inner())
  )
)

#outline(
  title: en([Figure Index]),
  target: figure.where(kind: image)
)
#pagebreak()

#outline(
  title: en([Table Index]),
  target: figure.where(kind: table)
)
#pagebreak()
