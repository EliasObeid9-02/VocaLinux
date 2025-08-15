// Custom captions
#let in_outline = state("in_outline", false)

#let flex_captions(long, short) = context if in_outline.get() { short } else { long }

// Text function
#let en_clean(body) = text(lang: "en", font: "Times New Roman", hyphenate: false, body)
#let en_clean_std(body) = text(size: 13pt, en_clean(body))

#let en(body) = text(lang: "en", dir: ltr, font: "Times New Roman", body)
#let en_std(body) = text(size: 13pt, en(body))

#let ar(body) = text(lang: "ar", dir: rtl, font: "Simplified Arabic", body)
#let ar_std(body) = text(size: 14pt, ar(body))

// Heading show rule constructor
#let numbered_heading(
  it, // The original heading object
  size, // The font size
  lang, // Document language
  above: 2.5em, // Default spacing above
  below: 0.5em, // Default spacing below
) = {
  set text(size: size, weight: "bold")

  if lang == "ar" {
    let numbering = str(counter(heading).display()).split("-").rev().join("-")
    align(right, block(above: above, below: below, text(dir: rtl)[
        #en_clean(numbering)#h(0.5em)#it.body
      ])
    )
  } else {
    let numbering = str(counter(heading).display())
    align(left, block(above: above, below: below, text(dir: ltr)[
        #en_clean(numbering)#h(0.5em)#it.body
      ])
    )
  }
}
