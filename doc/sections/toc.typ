#set outline.entry(fill: " ")

#show outline.entry: entry => {
  let prefix_text = entry.prefix().at("text")
  let body = text(
    lang: "ar",
    prefix_text.slice(0, prefix_text.len() - 1) + " " + entry.inner()
  )

  link(
    entry.element.location(),
    pad(right: entry.level * 1em, body)
  )
}

#align(center, outline(title: "الفهرس"))
#pagebreak()


#align(center, outline(title: "فهرس الأشكال", target: figure.where(kind: image)))
#pagebreak()


#align(center, outline(title: "فهرس الجداول", target: figure.where(kind: table)))
#pagebreak()
