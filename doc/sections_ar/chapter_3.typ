#import "../helpers.typ": flex_captions, ar, ar_std, en, en_std, en_clean, en_clean_std

= #ar([الفصل الثالث: التطبيق العملي])

#ar_std([
  يشرح هذا الفصل بالتفصيل تصميم النظام وتنفيذه، ويغطي البنية الهندسية وتدريب النموذج وتكامل المكونات الأساسية.
])

== #ar([بيئة التنفيذ ومجموعة التقنيات المستخدمة])

#ar_std([
  تم تطوير المشروع بالكامل بلغة البرمجة #en_clean_std([*Python*])، مستفيدًا من نظامها البيئي الشامل للحوسبة العلمية والتعلم العميق. يعتمد التنفيذ على مجموعة من المكتبات القوية مفتوحة المصدر، الملخصة في الجدول أدناه.

  #figure(
    table(
      columns: (4fr, 1fr, 1fr),
      align: (right+horizon, center+horizon, right+horizon),
      [*الاستخدام في المشروع*], [*المكتبة/الأداة*], [*الفئة*],

      [تستخدم لبناء وتدريب وتقييم نموذج #en_clean_std([LAS]) الأساسي. توفر طبقات أساسية مثل `Bidirectional` و`LSTM` و`Dense` لبناء بنية الشبكة.],
      [`TensorFlow` + `Keras`],
      [التعلم العميق],

      [الحزمة الأساسية للعمليات الحسابية الكفؤة ومعالجة مصفوفات البيانات، المستخدمة في جميع مراحل معالجة البيانات.],
      [`NumPy`],
      [رقمي وصوتي],

      [تستخدم لقراءة وكتابة ملفات صوتية بتنسيق `flac`، وتحويل البيانات الصوتية الخام إلى تنسيق رقمي مناسب للمعالجة.],
      [`SoundFile`],
      [رقمي وصوتي],

      [يستخدم لقياس أداء النموذج النهائي. يوفر تطبيقًا موحدًا لحساب معدل الخطأ في الكلمات (#en_clean_std([WER])) ومعدل الخطأ في الأحرف (#en_clean_std([CER])).],
      [`Jiwer`],
      [التقييم],

      [يُستخدم لعرض مقاييس التدريب والتحقق، مثل الخسارة ومعدل الأخطاء المكتوبة (#en_clean_std([WER]))، لإظهار أداء النموذج وتوجيه القرارات أثناء عملية التطوير التجريبي.],
      [`Matplotlib`],
      [العرض]
    ),
    kind: table,
    caption: flex_captions(
      [ملخص للمكتبات والأطر الأساسية مفتوحة المصدر المستخدمة لبناء النظام.],
      [مجموعة التقنيات المستخدمة في المشروع]
    )
  )
])

== #ar([تطبيق مجموعة البيانات وتنظيم المفردات])

#ar_std([
  يعتمد هذا النموذج على مجموعة #en_clean_std([*LibriSpeech*]). يوضح هذا القسم بالتفصيل المجموعات الفرعية المحددة المستخدمة، واستراتيجية التقسيم المستخدمة، وخطوات المعالجة المسبقة الشاملة التي تم تطويرها لتحويل البيانات الأولية إلى تنسيق مناسب لتدريب الشبكة العصبية.
])

=== #ar([اختيار البيانات وتقسيمها])

#ar_std([
  بالنسبة لهذا المشروع، تم اختيار مجموعة فرعية مركزة من مجموعة #en_clean_std([LibriSpeech]) لتحقيق التوازن بين وقت التدريب وأداء النموذج.
  - *مجموعات التدريب والتحقق*: تم استخدام تقسيم `train-clean-100`، الذي يحتوي على #en_clean_std([$100$]) ساعة من الكلام عالي الجودة، كمصدر أساسي لبيانات التدريب. من أجل مراقبة #en_clean_std([overfitting]) بشكل فعال وتوجيه عملية التدريب، تم تقسيم هذه المجموعة بشكل إضافي: تم تخصيص #en_clean_std([$80%$]) من البيانات لتدريب النموذج، بينما تم حجز الـ #en_clean_std([20%]) المتبقية كمجموعة للتحقق.
  - *مجموعة الاختبار*: لتوفير تقييم نهائي غير متحيز لأداء النموذج على البيانات غير المرئية، تم استخدام التقسيم الرسمي `test-clean`.

  أثناء التدريب، يتم تحميل البيانات على دفعات من #en_clean_std([$32$]) عينة. يتم خلط مجموعة التدريب في بداية كل حقبة لضمان عدم تعلم النموذج ترتيب الأمثلة، مما يعزز التعميم بشكل أفضل.
])

=== #ar([مسار المعالجة المسبقة])

#ar_std([
  تم تطبيق خط أنابيب معالجة مُسبقة مؤلف من مرحلتين لمعالجة البيانات الصوتية والنصية على التوالي.
])

#pagebreak()

==== #ar([معالجة الصوت: مخططات طيف #en_clean([Log-Mel])])

#ar_std([
  يتم تحويل أشكال الموجات الصوتية الخام إلى *مخططات طيفية #en_clean_std([Log-Mel])*، وهي تمثيل للميزات يحاكي الإدراك السمعي البشري ويكون فعالاً للغاية في مهام التعرف على الكلام. يتبع هذا التحويل تسلسل دقيق من الخطوات:
  + *التحميل*: يتم قراءة جميع الملفات الصوتية وتوحيدها بمعدل عينات يبلغ #en_clean_std([$16000$]) هرتز.
  + #en_clean_std([*STFT*]): يتم تطبيق تحويل فورييه قصير المدى (#en_clean_std([STFT])) على الإشارة باستخدام نافذة هان، وحجم تحويل فورييه سريع (#en_clean_std([FFT])) يبلغ #en_clean_std([$1024$])، وطول قفزة يبلغ #en_clean_std([$512$]). وهذا يحلل محتوى تردد الصوت بمرور الوقت.
  + *تحويل مقياس #en_clean_std([Mel])*: يتم تعيين الطيف الناتج إلى مقياس #en_clean_std([Mel]) باستخدام مجموعة مرشحات من #en_clean_std([$100$]) صندوق #en_clean_std([Mel]).
  + *التقييس اللوغاريتمي*: أخيرًا، يتم تطبيق دالة لوغاريتمية على مقاييس مخطط #en_clean_std([Mel]) الصوتي. وهذا يؤدي إلى ضغط النطاق الديناميكي للسمات، مما يؤدي إلى استقرار وتسريع تدريب النموذج.
])

==== #ar([معالجة النصوص وتنظيم المفردات])

#ar_std([
  تتم معالجة النصوص المقابلة لإنشاء تسلسلات رقمية مستهدفة لمفكك التشفير الخاص بالنموذج.
  + *المفردات*: تم تحديد مفردات ثابتة على مستوى الأحرف لضمان التناسق. وهي تتكون من #en_clean_std([$32$]) رمزًا فريدًا: #en_clean_std([$26$]) حرفًا إنجليزيًا صغيرًا، وعلامة اقتباس (`'`)، وحرف فراغ، وأربعة رموز خاصة (`<pad>`، `<unk>`، `<sos>`، `<eos>`).
  + *الترميز والتوحيد*: يتم تحويل كل نص مدون إلى أحرف صغيرة. ثم يتم ترميز كل نص مدون إلى تسلسل من الأحرف المكونة له.
  + *الرموز الخاصة*: للإشارة إلى بداية ونهاية التسلسل للمفكك، يتم إضافة الرموز `<sos>` (بداية الجملة) و`<eos>` (نهاية الجملة) إلى كل تسلسل رمزي، على التوالي.
  + *تعيين الأعداد الصحيحة*: يتم تعيين كل حرف في التسلسل النهائي إلى معرفه الصحيح الفريد من المفردات المحددة مسبقًا. يتم تعيين أي حرف غير موجود في المفردات إلى معرف الرمز `<unk>` (غير معروف). يتم استخدام الرمز `<pad>` لملء جميع التسلسلات داخل الدفعة بطول موحد.
])

== #ar([بنية النموذج وتطبيقه])

#ar_std([
  جوهر هذا المشروع هو تطبيق مخصص لهيكلية الاستماع والمتابعة والتهجئة (#en_clean_std([LAS]))، التي تم وصف أساسها النظري في الفصل الثاني. تم بناء النموذج باستخدام منظومتي `TensorFlow` و `Keras`. يوضح هذا القسم بالتفصيل الطبقات والمعلمات وخيارات التصميم المحددة التي تم اتخاذها لتطبيقنا.
])

=== #ar([تطبيق المستمع (المشفّر)])

#ar_std([
  وفقًا للهيكل الهرمي #en_clean_std([BLSTM])، تم تصميم #en_clean_std([Listener]) لمعالجة مخطط الطيف الصوتي #en_clean_std([log-Mel]) المدخل وإنتاج تمثيل ميزات مدمج وعالي المستوى. يتم ترتيب الطبقات على النحو التالي:
  + *طبقة #en_clean_std([BLSTM]) المدخلة*: تتم معالجة المدخلات أولاً بواسطة طبقة #en_clean_std([LSTM]) ثنائية الاتجاه (#en_clean_std([BLSTM])) قياسية تحتوي على #en_clean_std([$256$]) وحدة في كل اتجاه.
  + *طبقة التطبيع*: يتم تطبيع الناتج لتثبيت التدريب.
  + *مكدس #en_clean_std([BLSTM]) هرمي (#en_clean_std([pBLSTM]))*: يتم إدخال الناتج المطبع في مكدس من ثلاث طبقات #en_clean_std([pBLSTM]). تقلل كل طبقة عدد الاقسام الزمنية بمعامل اثنين وتستخدم #en_clean_std([$256$]) وحدة #en_clean_std([LSTM]) لكل اتجاه.
  
  الناتج النهائي لـ #en_clean_std([Listener]) هو تسلسل من متجهات الميزات التي تكون أقصر بـ #en_clean_std([$8$]) مرات في البعد الزمني من الطيف الصوتي الأصلي المدخل.
])

=== #ar([تطبيق المتهجئ (فك التشفير)])

#ar_std([
  برنامج #en_clean_std([Speller]) الخاص بنا هو جهاز فك تشفير قائم على الانتباه مصمم لإنشاء التدوين النهائي للنص. ويستفيد تطبيقه من العديد من المفاهيم المتقدمة الموضحة في الفصل الثاني.
  + *طبقة #en_clean_std([Character Embedding])*: يتم تحويل المعرف الصحيح للحرف الذي تم إنشاؤه مسبقًا إلى متجه كثيف ذي أبعاد #en_clean_std([$256$]).
  + *مكدس #en_clean_std([LSTM]) للمفكك*: تتم معالجة #en_clean_std([character embedding]) بواسطة مكدس من طبقتين #en_clean_std([LSTM]) أحاديتي الاتجاه، كل منها تحتوي على #en_clean_std([$512$]) وحدة. لمكافحة #en_clean_std([overfitting])، يتم تطبيق كل من `dropout` و`recurrent_dropout`.
  + *الانتباه متعدد الرؤوس المدرك للموقع*: في كل خطوة، يتم حساب متجه السياق باستخدام آلية *الانتباه المدرك للموقع* مع *#en_clean_std([$4$]) رؤوس*. وهذا يسمح للنموذج بالتركيز على أجزاء مختلفة من تمثيل الصوت في وقت واحد مع منعه من فقدان مكانه في التسلسل. البعد الداخلي لآلية الانتباه هو #en_clean_std([$512$]).
  + *طبقة توزيع الأحرف*: يتم تمرير الناتج من وحدة فك التشفير #en_clean_std([LSTM]) عبر طبقة `Dense` نهائية مع تابع تنشيط #en_clean_std([*Softmax*])، مما ينتج توزيع احتمالي على مفردات الأحرف.
])

== #ar([التجارب والنتائج])

#ar_std([
  يوضح هذا القسم بالتفصيل العملية التجريبية المستخدمة لتدريب النموذج وتقييمه. ويحدد التكوين العام للتدريب قبل عرض الإعدادات المحددة والنتائج لكل مرحلة تجريبية.
])

=== #ar([ إعدادات التدريب])

#ar_std([
  بناءً على المبادئ الموضحة في الفصل الثاني، تم استخدام مجموعة ثابتة من الأدوات:
  - *المُحسّن*: تم تدريب النموذج باستخدام مُحسّن #en_clean_std([*Adam*]). كما تم تطبيق #en_clean_std([*Gradient clipping*]) لمنع #en_clean_std([exploding gradients]).
  - *دالة الخسارة*: تم تطبيق دالة #en_clean_std([*Sparse Categorical Cross-Entropy*]) مخصصة للعمل مع مخرجات #en_clean_std([softmax]) للنموذج ولتجاهل الرموز `<pad>` تلقائيًا.
  - *مقاييس التقييم*: تمت مراقبة أداء النموذج باستخدام خسارة التدريب ودقة كل حرف ومعدل خطأ الحرف (#en_clean_std([CER])) ومعدل خطأ الكلمة (#en_clean_std([WER])).
])

=== #ar([استراتيجية #en_clean([Scheduled Sampling])])

#ar_std([
  بناءً على استراتيجيات #en_clean_std([scheduled sampling]) القياسية التي تمت مناقشتها في الفصل الثاني، يطبق هذا المشروع استراتيجية مخصصة تتميز بدورات خطية معدلة للتصعيد والاستقرار. تم تصميم هذا النهج لسد الفجوة بين التدريب والاستدلال مع الحفاظ على استقرار النموذج قدر الإمكان.

  #pagebreak()

  مراحل هذه الاستراتيجية هي:
  + *الإحماء الأولي*: تم استخدام فترة إحماء أولية، تم خلالها الحفاظ على ثبات احتمالية أخذ العينات. سمح ذلك للنموذج بالاستقرار قبل تعرضه لتنبؤاته الخاصة، التي قد تكون مشوشة.
  + *دورات التصعيد والاستقرار*: بعد الإحماء، استمر التدريب في دورات. يتم زيادة الاحتمال تدريجياً على مدى عدد محدد من الحقب الزمنية، تلاها فترة استقرار حيث بقي الاحتمال ثابت. سمحت هذه العملية التكرارية للنموذج بالتكيف مع الصعوبة المتزايدة دون أن يصبح غير مستقر.

  يتم تحديث الاحتمال، #en_clean_std([$epsilon$])، في بداية كل حقبة خلال مرحلة التصعيد. يتم حساب الاحتمال الجديد للحقبة الحالية، #en_clean_std([$epsilon_e$])، عن طريق إضافة زيادة خطية ثابتة إلى الاحتمال من الحقبة السابقة، #en_clean_std([$epsilon_(e-1)$]).

  يمكن التعبير عن قاعدة التحديث هذه على النحو التالي:

  #math.equation(
    [
      $epsilon_e = epsilon_(e-1) + (epsilon_f - epsilon_s) / E_r$
    ],
    block: true
  )

  حيث:
  - #en_clean_std([$epsilon_e$]): الاحتمال للحقبة الحالية #en_clean_std([$e$]).
  - #en_clean_std([$epsilon_(e-1)$]): الاحتمال في الحقبة السابقة.
  - #en_clean_std([$epsilon_s$]): الاحتمال الابتدائي في بداية مرحلة التصعيد.
  - #en_clean_std([$epsilon_f$]): الاحتمال النهائي المستهدف في نهاية مرحلة التصعيد.
  - #en_clean_std([$E_r$]): إجمالي عدد الحقبات التي يتم خلالها زيادة الاحتمال.

  ثم يتم ضبط القيمة الناتجة لضمان عدم تجاوزها الاحتمال النهائي المستهدف، #en_clean_std([$epsilon_f$]). كما أنها تتعامل بشكل صحيح مع الحالة التي يكون فيها #en_clean_std([$epsilon_f < epsilon_s$]) وتقلل الاحتمال دون التسبب في أخطاء.
])

=== #ar([المرحلة الأولى: التدريب الأولي])

#ar_std([
  ركزت المرحلة الأولى من التدريب، التي أجريت على مدى #en_clean_std([$240$]) حقبة، على تأسيس أداء ابتدائي. خلال هذه المرحلة، تم تكوين البنية باستخدام آلية #en_clean_std([*Multi-Head Attention*]) القياسية، بدون المكون الذي يراعي الموقع. كان الهدف الأساسي هو ضمان قدرة النموذج على تعلم المهمة الأساسية المتمثلة في تدوين الكلام وإيجاد مجموعة مستقرة من #en_clean_std([hyperparameters]).

  يوضح الجدول أدناه تفاصيل #en_clean_std([hyperparameters]) الرئيسية المستخدمة خلال هذه المرحلة.

  #figure(
    table(
      columns: (1fr, 1fr, 2fr),
      align: (center, center, left),
      [*القيمة النهائية*], [*القيمة الأولية*], en_clean_std([*Hyperparameter*]),
      en_clean_std([$0.0003$]), en_clean_std([$0.001$]), en_clean_std([Learning Rate]),
      en_clean_std([$2.5$]), en_clean_std([$5.0$]), en_clean_std([Gradient Clipping Norm]),
      en_clean_std([$0.5$]), en_clean_std([$0.05$]), en_clean_std([Sampling Probability (`epsilon`)])
    ),
    kind: table,
    caption: flex_captions(
      [قيم كل من #en_clean_std([Learning Rate]) و #en_clean_std([Gradient Clipping Norm]) و #en_clean_std([Sampling Probability]) خلال المرحلة الأولى من التدريب.],
      [قيم #en_clean_std([Hyperparameters]) في المرحلة الأولى]
    )
  )

  توضح الرسوم البيانية التالية تطور قيم #en_clean_std([metrics]) على مدار #en_clean_std([$240$]) حقبة من التدريب في المرحلة الأولى:

  #grid(
    columns: 2,
    gutter: 4pt,
    // Figure 1: Model Loss
    figure(
      image("../media/Training_Phase-1_Model-Loss.png", width: 100%),
      kind: image,
      caption: flex_captions(
        [قيم #en_clean_std([loss]) للنموذج على مدار #en_clean_std([$240$]) حقبة من تدريب المرحلة الأولى.],
        [قيم #en_clean_std([loss]) للنموذج في المرحلة الأولى]
      )
    ),

    // Figure 2: Model Accuracy
    figure(
      image("../media/Training_Phase-1_Model-Accuracy.png", width: 100%),
      kind: image,
      caption: flex_captions(
        [دقة كل حرف خلال التدريب في المرحلة الأولى.],
        [دقة النموذج في المرحلة الأولى]
      )
    ),

    // Figure 3: Character Error Rate (CER)
    figure(
      image("../media/Training_Phase-1_Model-CER.png", width: 100%),
      kind: image,
      caption: flex_captions(
        [معدل خطأ الحرف (#en_clean_std([CER])) خلال التدريب في المرحلة الأولى.],
        [قيم #en_clean_std([CER]) في المرحلة الأولى]
      )
    ),

    // Figure 4: Word Error Rate (WER)
    figure(
      image("../media/Training_Phase-1_Model-WER.png", width: 100%),
      kind: image,
      caption: flex_captions(
        [معدل الخطأ في الكلمات (#en_clean_std([WER])) خلال التدريب في المرحلة الأولى.],
        [قيم #en_clean_std([WER]) في المرحلة الأولى]
      )
    )
  )

  يوضح الجدول التالي عينة من التدوينات من نقاط مختلفة خلال هذه المرحلة، مما يوضح تقدم النموذج في التعلم. عينة التدوين هي #en_clean_std([\"do we really know the mountain well when we are not acquainted with the cavern\"]).

  #figure(
    table(
      columns: (1fr, 3fr, 8fr),
      align: (center+horizon, center+horizon, center+horizon),
      inset: 8pt,
      [*حقبة*], [*نوع الخرج*], [*النص*],

      table.cell(rowspan: 2, en_clean_std([$15$])),
      en_clean_std[Sampled], en_clean_std[di we really know the mountain well when we are not acquainted with the cave n],
      en_clean_std([Non-Sampled]), en_clean_std([```text 'aaa  wweeeeddiinneeeddiinneeeddii  wwooddeer        aann          asssssss  aann             aann        asssssss  aann             hhettiinnsss        asssssss  aann             aann          hheee             aann        hhettiinnsss        asssssss  aann             aann   '```]),

      table.cell(rowspan: 2, en_clean_std([$120$])),
      en_clean_std([Sampled]), en_clean_std([do we really know thatmountain well ween we are not accuainted tith aaa cavern]),
      en_clean_std([Non-Sampled]), en_clean_std([```text 'tee  rrr                                                                       '```]),

      table.cell(rowspan: 2, en_clean_std([$240$])),
      en_clean_std([Sampled]), en_clean_std([do ye rellly koow the mountain well whon we are not a quainted mith the cavern]),
      en_clean_std([Non-Sampled]), en_clean_std([```text 'doo   rrellly  ooww tthe  nnn tto wwell wwill  werre nntttt   qquiint  ff miicc  ccaareen'```]),
    ),
    kind: table,
    caption: flex_captions(
      [مقارنة بين نتائج النموذج لنفس العينة الصوتية في مراحل زمنية مختلفة خلال المرحلة الأولى من التدريب.],
      [مقارنة للنتائج في المرحلة الأولى]
    )
  )

  تُظهر النتائج الواردة في الجدول بوضوح التحسينات المستمرة لنوع الخرج #en_clean_std([non-sampled]) بمرور الوقت. في البداية كانت المخرجات عشوائية وغير مفهومة، ولكن في النهاية بدأت أنماط تركيب كلمات منطقية تظهر بشكل واضح مما يشير إلى أن النموذج بدأ يتعلم التعرف على الكلمات. من ناحية أخرى، تدهورت المخرجات من نوع #en_clean_std([sampled]) مع زيادة احتمال أخذ العينات لكنها ما تزال مفهومة.
])

#pagebreak()

=== #ar([المرحلة الثانية: تعزيز الانتباه من خلال إدراك الموقع])

#ar_std([
  أثبتت نتائج المرحلة الأولى وجود أساس متين، ولكنها سلطت الضوء أيضًا على تحدٍ رئيسي: واجهت تقنية #en_clean_std([Multi-Head Attention]) القياسية صعوبة في التعامل مع التسلسلات الصوتية الطويلة، حيث كانت تفقد مكانها أحيانًا أو تفشل في تدوين الكلام بالكامل. كان الهدف من المرحلة الثانية هو معالجة هذه المشكلة عن طريق استبدال آلية الانتباه القياسية بآلية #en_clean_std([*Location-Aware Attention*]) الأكثر قوة.

  تطلب هذا التغيير المعماري استمرار عملية التدريب بعناية. للسماح للنموذج بالتكيف، تمت إعادة تهيئة أوزان الطبقات الفرعية الجديدة للانتباه. ثم تم تدريب النموذج لمدة #en_clean_std([$140$]) حقبة إضافية.

  لمساعدة النموذج على التكيف مع البنية الجديدة، تمت إعادة تعيين احتمال #en_clean_std([scheduled sampling]) إلى قيمة أقل ثم تم زيادتها تدريجياً خلال فترة التدريب. تم الحفاظ على معدل التعلم وقيم #en_clean_std([gradient clipping]) من نهاية المرحلة السابقة.

  #figure(
    table(
      columns: (1fr, 1fr, 2fr),
      align: (center, center, left),
      [*القيمة النهائية*], [*القيمة الأولية*], en_clean_std([*Hyperparameter*]),
      en_clean_std([$0.0003$]), en_clean_std([$0.0003$]), en_clean_std([Learning Rate]),
      en_clean_std([$2.5$]), en_clean_std([$2.5$]), en_clean_std([Gradient Clipping Norm]),
      en_clean_std([$0.5$]), en_clean_std([$0.1$]), en_clean_std([Sampling Probability (`epsilon`)])
    ),
    kind: table,
    caption: flex_captions(
      [قيم كل من #en_clean_std([Learning Rate]) و #en_clean_std([Gradient Clipping Norm]) و #en_clean_std([Sampling Probability]) خلال المرحلة الثانية من التدريب.],
      [قيم #en_clean_std([Hyperparameters]) في المرحلة الثانية]
    )
  )

  #pagebreak()

  توضح الأشكال التالية تطور المقاييس على مدار #en_clean_std([$140$]) حقبة من التدريب في المرحلة الثانية.

  #grid(
    columns: 2,
    gutter: 4pt,
    // Figure 1: Model Loss
    figure(
      image("../media/Training_Phase-2_Model-Loss.png", width: 100%),
      kind: image,
      caption: flex_captions(
        [قيم #en_clean_std([loss]) للنموذج على مدار #en_clean_std([$140$]) حقبة من تدريب المرحلة الثانية.],
        [قيم #en_clean_std([loss]) للنموذج في المرحلة الثانية]
      )
    ),

    // Figure 2: Model Accuracy
    figure(
      image("../media/Training_Phase-2_Model-Accuracy.png", width: 100%),
      kind: image,
      caption: flex_captions(
        [دقة كل حرف خلال التدريب في المرحلة الثانية.],
        [دقة النموذج في المرحلة الثانية]
      )
    ),

    // Figure 3: Character Error Rate (CER)
    figure(
      image("../media/Training_Phase-2_Model-CER.png", width: 100%),
      kind: image,
      caption: flex_captions(
        [معدل خطأ الحرف (#en_clean_std([CER])) خلال التدريب في المرحلة الثانية.],
        [قيم #en_clean_std([CER]) في المرحلة الثانية]
      )
    ),

    // Figure 4: Word Error Rate (WER)
    figure(
      image("../media/Training_Phase-2_Model-WER.png", width: 100%),
      kind: image,
      caption: flex_captions(
        [معدل الخطأ في الكلمات (#en_clean_std([WER])) خلال التدريب في المرحلة الثانية.],
        [قيم #en_clean_std([WER]) في المرحلة الثانية]
      )
    ),
  )

  كما يتضح النجاح النوعي لهذا التغيير المعماري في مقاييس الأداء. خلال هذه المرحلة، تجاوز كل من #en_clean_std([CER]) و #en_clean_std([WER]) أفضل القيم المسجلة في المرحلة الأولى، محققين أرقامًا قياسية جديدة للنموذج.

  #pagebreak()

  توضح الجداول التالية عدة أمثلة للتدوين من نقاط مختلفة خلال هذه المرحلة، كما قمنا بتضمين مخرجات من نهاية المرحلة الأولى لتوضيح التحسينات التي تم تحقيقها من خلال تطبيق ”الانتباه المدرك للموقع“.

  #figure(
    table(
      columns: (1.5fr, 2.5fr, 8fr),
      align: (center+horizon, center+horizon, center+horizon),
      inset: 8pt,
      [*المرحلة/الحقبة*], [*نوع المخرجات*], [*النص*],

      [$1$/$240$], en_clean_std([Beam Search]), [```text 'whnn it is a queestion of proving alooond aass  ooiitty ssnnss ween hassaaddbbeen considered floweed to go toooo tthrr tto go ttotthe bottom'```],
      [$2$/$20$], en_clean_std([Beam Search]), [```text 'when iss accuushhddiin off rrogginng a ooon    gollf  a  so iide  since  wee  has aad bbeen considered faamm to go  to  ffrr tto ggo  to  hhe bottom'```],
      [$2$/$80$], en_clean_std([Beam Search]), [```text 'when it is accuusseed ee  ff rroiing awwoodd   gollf aass  syy t  since hhen has it been considered froww tto go to ffor tto go ttotthe botuum'```],
      [$2$/$140$], en_clean_std([Beam Search]), [```text 'when it is a question oof proping a ooond  ggolf aa  sosside    ssnnc  ween has i  been considered frowwntto go too far to go to the bathom'```]
    ),
    kind: table,
    caption: flex_captions(
      [مقارنة النتائج للجملة: #en_clean_std([\"when it is a question of probing a wound a gulf a society since when has it been considered wrong to go too far to go to the bottom\"]).],
      [مقارنة لجملة متوسطة الطول]
    )
  )

  من المقارنة نلاحظ أن النموذج أصبح أكثر مقاومة للأخطاء السابقة وأصبح الآن قادراً على إخراج الكلمات الصحيحة حتى بعد الأخطاء.

  #pagebreak()

  #figure(
    table(
      columns: (1fr, 3fr, 8fr),
      align: (center+horizon, center+horizon, center+horizon),
      inset: 8pt,
      [*المرحلة/الحقبة*], [*نوع المخرجات*], [*النص*],

      [$1$/$240$], en_clean_std([Beam Search]), [```text 'to keep aee loot annd to resk you frr   llivinn toowwhole ttee goll whhereiit by  ffragmments of somme llnnguunn wwhich man has spoken  nn  which wooeedoott ofwwhich ttss compoccated too eexteend  oftthe records oofssocial observatioon issseeltth'```],
      [$2$/$20$], en_clean_std([Beam Search]), [```text 'to  eee  lott annd too uus  you  ff   bbliivinn  oo  hhole abbove thhe goll  wheeriit buut iffrrggmeents oof some llnnguage wwhichmman has spokken oof wwhich  siveaalizaationnaas commosee  orrr  vve  whhich it iss complitatee  twweettsstnnd  of tthe recorrs  of sss'```],
      [$2$/$80$], en_clean_std([Beam Search]), [```text 'to keep a foo  and too rest you foo  a bivvi  in  nn  ooll tto oove hhe golf  whereiit but iffraagmennt  o  some language which man has spoken and whicch would other iise be lost that  is oo say one of the elements good or aad off whicch iivilizaatiooniis composed '```],
      [$2$/$140$], en_clean_std([Beam Search]), [```text 'to people fol  and to rescue froom a blivioo  nn  hhole to  oov  the golf wher  it but ifraagment oofssome language which man has spoken andwwhich woudd otherwiiee beloost mataas  oo say oone of tthe elements good or aad of whicc  sivilizatiooniis composed oor  by '```]
    ),
    kind: table,
    caption: flex_captions(
      [مقارنة النتائج للجملة: #en_clean_std([\"to keep afloat and to rescue from oblivion to hold above the gulf were it but a fragment of some language which man has spoken and which would otherwise be lost that is to say one of the elements good or bad of which civilization is composed or by which it is complicated to extend the records of social observation is to serve civilization itself\"]).],
      [مقارنة جملة طويلة الطول]
    )
  )

  مرة أخرى، توضح هذه الجملة الطويلة التحسينات التي حققها النموذج من خلال إدراك الموقع، على سبيل المثال، فقد فاتت النسخة المدونة في المرحلة الأولى الكثير من أنماط الكلمات مثل تلك الخاصة بـ #en_clean_std([\"above\"]) و #en_clean_std([\"civilization\"])، بينما في نهاية المرحلة الثانية، أنتج النموذج أنماط كلمات لهما.
])

=== #ar([المرحلة الثالثة: مكافحة #en_clean([Overfitting]) باستخدام #en_clean([Regularization])])

#ar_std([
  بعد التحسينات المعمارية في المرحلة الثانية، بدأت تظهر علامات #en_clean_std([overfitting]) بشكل واضح. في حين استمرت مقاييس التدريب في التحسن، بدأت مقاييس التحقق تستقر، مما يشير إلى أن النموذج بدأ في حفظ بيانات التدريب بدلاً من تعلم كيفية تعميم استنتاجاته. على الرغم من أن مقاييس التحقق أظهرت علامات استقرار طوال فترة التدريب، إلا أن السبب نُسب إلى قيمة #en_clean_std([sampling rate]) المنخفضة.

  كان الهدف من هذه المرحلة الموجزة المكونة من $55$ بؤرة تخطيطية هو مكافحة #en_clean_std([overfitting]) عن طريق استخدام #en_clean_std([regularization]). تمثلت الاستراتيجية الأساسية في زيادة معدلات #en_clean_std([dropout]) في وحدة فك التشفير وتطبيق #en_clean_std([dropout]) على وحدة التشفير لأول مرة.

  ولزيادة #en_clean_std([regularization])، تم رفع قيمتي #en_clean_std([\"dropout\"]) و #en_clean_std([\"recurrent_dropout\"]) في مكدس #en_clean_std([LSTM]) الخاص بالمستمع. بالإضافة إلى ذلك، تمت إضافة طبقة #en_clean_std([\"dropout\"]) جديدة إلى مكدس #en_clean_std([pBLSTM]) الخاص بالمستمع لتطبيق #en_clean_std([regularization]) على نتائج التشفير الداخلية. تم الحفاظ على #en_clean_std([hyperparameters]) الأخرى من نهاية المرحلة السابقة.

  #figure(
    table(
      columns: (1fr, 1fr, 2fr),
      align: (center, center, left),
      [*القيمة النهائية*], [*القيمة الأولية*], en_clean_std([*Hyperparameter*]),
      en_clean_std([$0.0003$]), en_clean_std([$0.0003$]), en_clean_std([Learning Rate]),
      en_clean_std([$2.5$]), en_clean_std([$2.5$]), en_clean_std([Gradient Clipping Norm]),
      en_clean_std([$0.5$]), en_clean_std([$0.1$]), en_clean_std([Sampling Probability (`epsilon`)]),
      en_clean_std([$0.3$]), en_clean_std([$0.17$]), en_clean_std([Decoder Dropout]),
      en_clean_std([$0.1$]), en_clean_std([$0.0$]), en_clean_std([Encoder Dropout]),
    ),
    kind: table,
    caption: flex_captions(
      [قيم كل من #en_clean_std([Learning Rate]) و #en_clean_std([Gradient Clipping Norm]) و #en_clean_std([Sampling Probability]) خلال المرحلة الثالثة من التدريب.],
      [قيم #en_clean_std([Hyperparameters]) في المرحلة الثالثة]
    )
  )

  تم تدريب النموذج على $55$ حقبة مع #en_clean_std([regularization]) محسّنة. توضح الأشكال التالية تطور المقاييس خلال هذه المرحلة.

  #grid(
    columns: 2,
    gutter: 4pt,
    // Figure 1: Model Loss
    figure(
      image("../media/Training_Phase-3_Model-Loss.png", width: 100%),
      kind: image,
      caption: flex_captions(
        [قيم #en_clean_std([loss]) للنموذج على مدار #en_clean_std([$140$]) حقبة من تدريب المرحلة الثالثة.],
        [قيم #en_clean_std([loss]) للنموذج في المرحلة الثالثة]
      )
    ),

    // Figure 2: Model Accuracy
    figure(
      image("../media/Training_Phase-3_Model-Accuracy.png", width: 100%),
      kind: image,
      caption: flex_captions(
        [دقة كل حرف خلال التدريب في المرحلة الثالثة.],
        [دقة النموذج في المرحلة الثالثة]
      )
    ),

    // Figure 3: Character Error Rate (CER)
    figure(
      image("../media/Training_Phase-3_Model-CER.png", width: 100%),
      kind: image,
      caption: flex_captions(
        [معدل خطأ الحرف (#en_clean_std([CER])) خلال التدريب في المرحلة الثالثة.],
        [قيم #en_clean_std([CER]) في المرحلة الثالثة]
      )
    ),

    // Figure 4: Word Error Rate (WER)
    figure(
      image("../media/Training_Phase-3_Model-WER.png", width: 100%),
      kind: image,
      caption: flex_captions(
        [معدل الخطأ في الكلمات (#en_clean_std([WER])) خلال التدريب في المرحلة الثالثة.],
        [قيم #en_clean_std([WER]) في المرحلة الثالثة]
      )
    ),
  )

  كان تحليل هذه المرحلة قاطعًا. فبينما استمرت مقاييس التدريب في إظهار استقرار أو تحسن طفيف، لم تظهر مقاييس التحقق أي علامات تقدم واضحة. فشل تحسن قيم #en_clean_std([WER]) و #en_clean_std([CER]) على بيانات التحقق مقارنة بالأرقام القياسية التي تم تحقيقها في نهاية المرحلة الثانية. نظرًا لأن زيادة #en_clean_std([regularization]) لم تسفر عن التحسن المطلوب في التعميم، تم إنهاء هذه المرحلة التجريبية. دفعت هذه النتيجة إلى تحول في الاستراتيجية نحو استكشاف حلول محتملة مختلفة.

  تمت تجربة العديد من المحاولات من أجل مكافحة مشكلة الإفراط في التركيب. من خلال مراجعة شاملة للورقة البحثية الأصلية، وكود النموذج، واستراتيجية التدريب المستخدمة، حددنا أن أحد أسباب المشكلة هو التباين بين حجم بيانات التدريب وحجم الشبكة. وعلى سبيل المقارنة، ذكرت الورقة البحثية أنهم قاموا بتدريب النموذج على بيانات تصل مدتها إلى #en_clean_std([$2000$]) ساعة باستخدام نفس حجم الشبكة المستخدمة في تطبيقنا، في حين أن تطبيقنا تم تدريبه على #en_clean_std([$100$]) ساعة فقط من بيانات التدريب.

  وقد دفعنا هذا التباين الكبير إلى التفكير في تقليل حجم الشبكة بشكل كبير، وتطبيق تعديلات كبيرة على البيانات من خلال طريقة #en_clean_std([SpecAugment])، واستخدام مجموعة بيانات أكبر مثل مجموعة #en_clean_std([$360$]) ساعة. ومن التقنيات الأخرى التي تمت تجربتها #en_clean_std([*Cyclical Learning Rate Schedule*]). تم تدريب نسخ النموذج التي تستخدم هذه التعديلات على مجموعة متنوعة من الحقبات تتراوح من #en_clean_std([$20$]) إلى #en_clean_std([$100$])، ولم تقدم معظم الإصدارات أي تحسينات أو علامات محتملة لحل #en_clean_std([overfitting]).
])

== #ar([ المفسر القائم على القواعد])

#ar_std([
  الناتج من نموذج #en_clean_std([ASR]) هو تدوين نصي خام، والذي، على الرغم من دقته في كثير من الأحيان، إلا أنه ليس أمرًا صحيحًا من الناحية النحوية في لغة #en_clean_std([shell]). اللغة المنطوقة غامضة بطبيعتها وتحتوي على تهجئات صوتية (مثل #en_clean_std([\"slash\"]) و #en_clean_std([\"dash\"])) لا يمكن لسطر الأوامر تنفيذها. المكون الأخير لهذا المشروع هو مفسر قائم على القواعد مصمم لسد هذه الفجوة. والغرض الوحيد منه هو تحليل النص الخام الناتج عن #en_clean_std([ASR]) وترجمته إلى أمر لينكس صحيح وقابل للتنفيذ.

  تتم عملية التفسير على مرحلتين رئيسيتين: أولاً، يقوم *المترجم* بتحويل الكلمات الرئيسية المنطوقة إلى الأحرف والأوامر المقابلة لها، وثانياً، يقوم *محلل المسار* بتصحيح الأخطاء الإملائية المحتملة في مسارات الملفات والمجلدات بشكل ذكي.
])

=== #ar([الترجمة ومفرداتها])

#ar_std([
  تدير صف `Translator` المرحلة الأولى من الترجمة. وهي تعمل على مفردات محددة مسبقًا من الكلمات الرئيسية التي تربط الكلمات المنطوقة بتمثيلها النصي المقصود. وتنقسم هذه المفردات إلى عدة فئات.
])

==== #ar([المفردات المعترف بها])

#ar_std([
  يتم تعريف مفردات المفسر بشكل صريح للتعامل مع الأوامر والأحرف المميزة والأرقام والأبجدية الصوتية للتهجئة.

  - *الأوامر*: يتم التعرف على مجموعة أساسية من أوامر لينكس الشائعة.
  - *أحرف مميزة*: يتم تعيين الكلمات المنطوقة مثل #en_clean_std([\"slash\"]) أو #en_clean_std([\"dot\"]) أو #en_clean_std([\"double quote\"]) إلى معادلاتها الرمزية (`/` و `.` و `"`).
  - *الأرقام*: يتم تحويل الأرقام المنطوقة (#en_clean_std([\"zero\"]) إلى #en_clean_std([\"nine\"])) إلى شكلها الرقمي.
  - *الأبجدية الصوتية*: يتم استخدام أبجدية صوتية قياسية (على سبيل المثال، #en_clean_std([\"adam\"]) ل #en_clean_std([\"a\"])، و #en_clean_std([\"boy\"]) ل #en_clean_std([\"b\"])) للسماح بتهجئة دقيقة لا غموض فيها لأسماء الملفات أو الوسائط.

  يوضح الجدول أدناه تفاصيل الكلمات المفتاحية الأساسية التي يتعرف عليها المفسر.

  #figure(
    table(
      columns: (2fr, 4fr),
      align: (right, center),
      [*التصنيف*], [*أمثلة*],
      [أوامر], en_std([`ls`, `cd`, `cp`, `mv`, `rm`, `mkdir`, `echo`, `touch`]),
      [أحرف مميزة], en_std([`slash` -> `/`, `hyphen` -> `-`, `double quote` -> `"`]),
      [أرقام], en_std([`one` -> `1`, `five` -> `5`, `nine` -> `9`]),
      [أبجدية صوتية], en_std([`adam` -> `a`, `robert` -> `r`, `zebra` -> `z`]),
    ),
    kind: table,
    caption: flex_captions(
      [التصنيفات والأمثلة من المفردات المعترف بها للمفسر الفوري.],
      [مفردات المفسر الفوري]
    )
  )
])

==== #ar([كشف الكلمات الدلالية متعددة الكلمات])

#ar_std([
  للتعامل مع الكلمات الدلالية التي تتكون من أكثر من كلمة واحدة (على سبيل المثال، #en_clean_std([\"double quote\"]))، يقوم المفسر بتطبيق خوارزمية أطول مطابقة أولاً. حيث يقوم بمسح نص الإدخال بحثًا عن أطول تسلسل ممكن للكلمات المفتاحية في أي موضع معين. على سبيل المثال، عند معالجة عبارة #en_clean_std([\"double quote my file\"])، سيتعرف أولاً على #en_clean_std([\"double quote\"]) كرمز رمزي واحد مكون من كلمتين ويترجمه إلى #en_clean_std([\"double quote\"])، بدلاً من معالجة #en_clean_std([\"double\"]) و #en_clean_std([\"quote\"]) ككلمتين منفصلتين غير معروفتين.
])

==== #ar([آلية تجاوز الكلمات الدلالية])

#ar_std([
  للتعامل مع الحالات التي تكون فيها الكلمة الدلالية مقصودة كوسيط حرفي، يتضمن النظام آلية تجاوز. تتسبب الكلمة الدليلية #en_clean_std([*backslash*]) التي تسبق أي كلمة الدليلية أخرى معترف بها في أن يعامل المفسر الكلمة الدليلية اللاحقة كنص عادي، متجاهلاً وظيفتها الخاصة. على سبيل المثال، ستؤدي عبارة #en_clean_std([\"echo space backslash space\"]) إلى الأمر #en_clean_std([\"echo space\"])، مما يؤدي إلى التعامل مع كلمة #en_clean_std([\"space\"]) على أنها كلمة حرفية بدلاً من مسافة فارغة بين الكلمات.
])

=== #ar([معالجة المسار وتصحيحه])

#ar_std([
  أحد المصادر الشائعة للخطأ في التعرف على الكلام هو الخطأ الإملائي في أسماء الملفات أو أسماء المجلدات. تم تصميم صف ’PathResolver‘ للتخفيف من هذه المشكلة عن طريق تصحيح المسارات بذكاء.

  بعد الترجمة الأولية، يقوم المفسر بتقسيم الأمر إلى أجزاء. يتم تمرير أي جزء يحتوي على فاصل مسار (`/`) إلى محلل المسار. ثم يقوم المحلل بمعالجة المسار جزءًا تلو الآخر. لكل جزء، يتحقق من المجلد المطابق على نظام الملفات ويستخدم خوارزمية مطابقة سلسلة ضبابية (`difflib.get_close_matches`) للعثور على أقرب اسم ملف أو مجلد مطابق. إذا تم العثور على تطابق قريب (مع حد تشابه يبلغ 0.5)، فإنه يستبدل المقطع الذي يحتمل أن يكون به خطأ إملائي بالاسم الصحيح.

  تعمل هذه الآلية لكل من المسارات المطلقة والنسبية، مما يزيد بشكل كبير من متانة النظام من خلال تصحيح أخطاء التدوين الطفيفة التي قد تتسبب في فشل الأمر النهائي.
])
