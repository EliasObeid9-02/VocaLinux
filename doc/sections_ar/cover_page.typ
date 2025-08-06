#import "../helpers.typ": ar, ar_std, en, en_std

#set page(footer: align(left, ar_std([2024-2025])))

#table(
  stroke: 0pt,
  rows: (3fr, 3fr, 2fr, 1fr),
  columns: (1fr, 1fr, 1fr),
  table.cell(
    colspan: 1,
    align(left,
     scale(75%, image("../media/Latakia-University_Logo.png"))
   )
  ),
  table.cell(
    colspan: 2,
    align(right, [
      #ar_std([الجمهورية العربية السورية])\
      #ar_std([وزارة التعليم العالي])\
      #ar_std([جامعة اللاذقية - كلية الهندسة المعلوماتية ])\
      #ar_std([قسم الذكاء الصنعي])
    ])
  ),

  table.cell(
    rowspan: 1,
    colspan: 3,
    align(center)[
      #heading(
        ar([تحويل الكلام إلى أوامر طرفية في لينكس]),
        level: 1,
        outlined: false,
        numbering: none
      )
      #heading(
        en([Transcribing Speech into Linux Terminal Commands]),
        level: 1,
        outlined: false,
        numbering: none
      )
      #ar_std([مشروع تخرج])
    ]
  ),

  table.cell(
    rowspan: 1,
    colspan: 3,
    align(center)[
      #ar_std([إعداد الطلاب])\
      #ar_std([الياس عبيد])\ \
      #ar_std([إشراف])\
      #ar_std([د. سامر سليمان])
    ]
  )
)

#pagebreak()
