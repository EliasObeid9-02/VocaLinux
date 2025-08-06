#import "../helpers.typ": en, en_std

#set page(footer: align(right, en_std([2024-2025])))

#table(
  stroke: 0pt,
  rows: (3fr, 3fr, 2fr, 1fr),
  columns: (1fr, 1fr, 1fr),

  table.cell(
    colspan: 2,
    align(left, text(lang: "en", size: 13pt)[
      #en_std([Syrian Arab Republic])\
      #en_std([Ministry of Higher Education])\
      #en_std([Latakia University - Faculty of Informatics Engineering])\
      #en_std([Department of Artificial Intelligence])
    ])
  ),
  table.cell(
    colspan: 1,
    align(right,
     scale(75%, image("../media/Latakia-University_Logo.png"))
   )
  ),

  table.cell(
    rowspan: 1,
    colspan: 3,
    align(center)[
      #heading(
        en([Transcribing Speech into Linux Terminal Commands]),
        level: 1,
        outlined: false,
        numbering: none
      )
      #en_std([Graduation Project])
    ]
  ),

  table.cell(
    rowspan: 1,
    colspan: 3,
    align(center)[
      #en_std([*Prepared By*])\
      #en_std([Elias Obeid])\ \
      #en_std([*Supervised By*])\
      #en_std([Dr.-Eng. Samer Sulaiman])
    ]
  )
)

#pagebreak()
