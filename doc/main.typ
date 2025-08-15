#import "helpers.typ": ar, ar_std, en, en_std, numbered_heading

// The desired language of the document, can be "en" or "ar".
#let lang = sys.inputs.at("lang", default: "en")

#set page(paper: "a4")

// Heading rules
#set heading(numbering: "1-1-1-1-1")
#show heading: set text(hyphenate: false)

// Rule for Level 1 Heading (unique style)
#show heading.where(level: 1): it => {
  set text(size: 28pt, weight: "bold")
  
  align(center, block(below: 1.0em, it.body))

  // Resetting figure counters
  counter(figure.where(kind: image)).update(0)
  counter(figure.where(kind: table)).update(0)
}

// Rules for Levels 2-5 now use the helper function
#show heading.where(level: 2): it => numbered_heading(it, 20pt, lang)
#show heading.where(level: 3): it => numbered_heading(it, 18pt, lang)
#show heading.where(level: 4): it => numbered_heading(it, 16pt, lang)
#show heading.where(level: 5): it => numbered_heading(it, 14pt, lang)

// Figure rules
#show figure.where(kind: image): set figure(
  supplement: if lang == "ar" {
    ar_std([*الشكل*])
  } else {
    en_std([*Figure*])
  },
  numbering: n => {
    if lang == "ar" {
      ar_std([
        *#numbering("1-1", counter(heading).get().first(), n)*
      ])
    } else {
      en_std([
        *#numbering("1-1", counter(heading).get().first(), n)*
      ])
    }
  }
)

#show figure.where(kind: table): set figure(
  supplement: if lang == "ar" {
    ar_std([*الجدول*])
  } else {
    en_std([*Table*])
  },
  numbering: n => {
    if lang == "ar" {
      ar_std([
        *#numbering("1-1", counter(heading).get().first(), n)*
      ])
    } else {
      en_std([
        *#numbering("1-1", counter(heading).get().first(), n)*
      ])
    }
  }
)

// Paragraph rule
#set par(
  leading: 0.75em,
  spacing: 2.0em,
  linebreaks: "optimized",
  justify: true
)


#if lang == "ar" {
  include "sections_ar/cover_page.typ"
  include "sections_ar/abstract.typ"
  include "sections_ar/toc.typ"

  // set page(numbering: "1", number-align: left+bottom, binding: right)
  set page(
    numbering: "1",
    binding: left,
    footer: align(center+top, ar_std(
      context counter(page).display())
    )
  )
  counter(page).update(1)
  include "sections_ar/chapter_1.typ"
  include "sections_ar/chapter_2.typ"
  include "sections_ar/chapter_3.typ"
} else {
  include "sections_en/cover_page.typ"
  include "sections_en/abstract.typ"
  include "sections_en/toc.typ"

  set page(
    numbering: "1",
    binding: left,
    footer: align(center+top, en_std(
      context counter(page).display())
    )
  )
  counter(page).update(1)
  include "sections_en/chapter_1.typ"
  include "sections_en/chapter_2.typ"
  include "sections_en/chapter_3.typ"
}
