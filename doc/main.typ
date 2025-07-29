#import "helpers.typ": ar, ar_std, en, en_std

// The desired language of the document, can be "en" or "ar".
#let lang = sys.inputs.at("lang", default: "ar")

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
  supplement: if lang == "ar" {
    ar([*الشكل*])
  } else {
    en([*Figure*])
  },
  numbering: n => {
    if lang == "ar" {
      ar([
        *#str(numbering("1-1", counter(heading).get().first(), n)).rev()*
      ])
    } else {
      en([
        *#str(numbering("1-1", counter(heading).get().first(), n))*
      ])
    }
  }
)

#show figure.where(kind: table): set figure(
  supplement: if lang == "ar" {
    ar([*الجدول*])
  } else {
    en([*Table*])
  },
  numbering: n => {
    if lang == "ar" {
      ar([
        *#str(numbering("1-1", counter(heading).get().first(), n)).rev()*
      ])
    } else {
      en([
        *#str(numbering("1-1", counter(heading).get().first(), n))*
      ])
    }
  }
)


#if lang == "ar" {
  include "sections_ar/cover_page.typ"
  include "sections_ar/abstract.typ"
  include "sections_ar/toc.typ"

  set page(numbering: "1", number-align: left+bottom, binding: right)
  counter(page).update(1)
  include "sections_ar/chapter_1.typ"
  include "sections_ar/chapter_2.typ"
  include "sections_ar/chapter_3.typ"
} else {
  include "sections_en/cover_page.typ"
  include "sections_en/abstract.typ"
  include "sections_en/toc.typ"

  set page(numbering: "1", number-align: right+bottom, binding: left)
  counter(page).update(1)
  include "sections_en/chapter_1.typ"
  include "sections_en/chapter_2.typ"
  include "sections_en/chapter_3.typ"
}
