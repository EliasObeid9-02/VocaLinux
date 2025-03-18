#set page(paper: "a4")

#show text.where(lang: "ar"): body => {
  set text(font: "Simplified Arabic")
  body
}

#show text.where(lang: "en"): body => {
  set text(font: "Times New Roman")
  body
}

#include "sections/cover_page.typ"
#include "sections/abstract.typ"
#include "sections/chapter_1.typ"
#include "sections/chapter_2.typ"
#include "sections/chapter_3.typ"
