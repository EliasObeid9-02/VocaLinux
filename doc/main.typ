// The desired language of the document, can be "en" or "ar".
#let lang = sys.inputs.at("lang", default: "en")

#set page(paper: "a4")

// Heading rules
#set heading(numbering: "1-1-1-1-1")

#show heading.where(level: 1): it => {
  set text(size: 28pt, weight: "bold")
  align(center, it.body)

  // Resetting figure counters
  counter(figure.where(kind: image)).update(0)
  counter(figure.where(kind: table)).update(0)
}

#show heading.where(level: 2): it => {
  set text(size: 20pt, weight: "bold")
  if lang == "ar" {
    let numbering = str(counter(heading).display()).rev()
    align(right, text(dir: rtl)[
      #numbering#h(0.25em)#it.body
    ])
  } else {
    let numbering = str(counter(heading).display())
    align(left, text(dir: ltr)[
      #numbering#h(0.25em)#it.body
    ])
  }
}

#show heading.where(level: 3): it => {
  set text(size: 18pt, weight: "bold")
  if lang == "ar" {
    let numbering = str(counter(heading).display()).rev()
    align(right, text(dir: rtl)[
      #numbering#h(0.25em)#it.body
    ])
  } else {
    let numbering = str(counter(heading).display())
    align(left, text(dir: ltr)[
      #numbering#h(0.25em)#it.body
    ])
  }
}

#show heading.where(level: 4): it => {
  set text(size: 16pt, weight: "bold")
  if lang == "ar" {
    let numbering = str(counter(heading).display()).rev()
    align(right, text(dir: rtl)[
      #numbering#h(0.25em)#it.body
    ])
  } else {
    let numbering = str(counter(heading).display())
    align(left, text(dir: ltr)[
      #numbering#h(0.25em)#it.body
    ])
  }
}

#show heading.where(level: 5): it => {
  set text(size: 14pt, weight: "bold")
  if lang == "ar" {
    let numbering = str(counter(heading).display()).rev()
    align(right, text(dir: rtl)[
      #numbering#h(0.25em)#it.body
    ])
  } else {
    let numbering = str(counter(heading).display())
    align(left, text(dir: ltr)[
      #numbering#h(0.25em)#it.body
    ])
  }
}

// Figure rules
#show figure.where(kind: image): set figure(
  supplement: text(weight: "bold")[Figure],
  numbering: n => {
    text(weight: "bold", numbering("1-1", counter(heading).get().first(), n))
  },
)

#show figure.where(kind: table): set figure(
  supplement: text(weight: "bold")[Table],
  numbering: n => {
    text(weight: "bold", numbering("1-1", counter(heading).get().first(), n))
  },
)


#if lang == "ar" {
  set text(lang: "ar", dir: rtl, font: "Simplified Arabic", size: 14pt)
  show text.where(lang: "en"): it => {
    set text(lang: "en", dir: ltr, font: "Times New Roman", size: 13pt)
    it
  }

  include "sections_ar/cover_page.typ"
  include "sections_ar/abstract.typ"
  include "sections_ar/toc.typ"

  set page(numbering: "1", number-align: left+bottom)
  counter(page).update(1)
  include "sections_ar/chapter_1.typ"
  include "sections_ar/chapter_2.typ"
  include "sections_ar/chapter_3.typ"
} else {
  set text(lang: "en", dir: ltr, font: "Times New Roman", size: 13pt)

  include "sections_en/cover_page.typ"
  include "sections_en/abstract.typ"
  include "sections_en/toc.typ"

  set page(numbering: "1", number-align: right+bottom)
  counter(page).update(1)
  include "sections_en/chapter_1.typ"
  include "sections_en/chapter_2.typ"
  include "sections_en/chapter_3.typ"
}
