// Custom captions
#let in_outline = state("in_outline", false)

#let flex_captions(long, short) = context if in_outline.get() { short } else { long }

// Text function
#let en_clean(body) = text(lang: "en", font: "Times New Roman", body)
#let en_clean_std(body) = text(size: 13pt, en_clean(body))

#let en(body) = text(lang: "en", dir: ltr, font: "Times New Roman", body)
#let en_std(body) = text(size: 13pt, en(body))

#let ar(body) = text(lang: "ar", dir: rtl, font: "Simplified Arabic", body)
#let ar_std(body) = text(size: 14pt, ar(body))
