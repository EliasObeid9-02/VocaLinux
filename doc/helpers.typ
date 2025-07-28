// Custom captions
#let in_outline = state("in_outline", false)

#let flex_captions(long, short) = context if in_outline.get() { short } else { long }

// English text function
#let en(body) = text(lang: "en", body)
