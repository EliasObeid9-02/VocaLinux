#import "../helpers.typ": ar, ar_std, en, en_std

#heading(
  ar([ملخص]),
  level: 1,
  outlined: false,
  numbering: none,
)

#ar_std([
  تُعد واجهة الأوامر (#en_std([CLI])) أداة قوية للتفاعل مع أجهزة الحاسوب، ولكن قد يتعذر استخدامها من قبل المستخدمين الذين يعانون من إعاقات جسدية تجعل الكتابة صعبة أو مستحيلة. يهدف هذا المشروع إلى حل هذه المشكلة من خلال إنشاء نظام يترجم اللغة المنطوقة إلى أوامر لينكس قابلة للتنفيذ، مما يجعل واجهة الأوامر (#en_std([CLI])) أكثر سهولة وصالحة الاستخدام للجميع.

  الهدف الأساسي هو تطوير نظام مزدوج قوي يحول الكلام إلى أوامر بدقة. ويتم تحقيق ذلك من خلال دمج مكوّنين أساسيين: نموذج تعلّم عميق للتعرّف على الكلام ومترجم فوري حتمي لتوليد الأوامر. يعتمد مكوّن التعرّف على الكلام على نموذج "الاستماع والحضور والتهجئة" (#en_std([LAS]))، وهو مصمم لتحويل الكلمات المنطوقة إلى نص. ثم تتم معالجة هذا النص بواسطة مترجم فوري قائم على القواعد يقوم بتعيين عبارات اللغة الطبيعية إلى أوامر لينكس الصحيحة نحويًا.

  يتم تقييم فعالية النظام باستخدام مقياسين رئيسيين: معدل الخطأ في الكلمات (#en_std([WER])) لقياس دقة نموذج نسخ الكلام، ودقة الأمر الكلي لتقييم أداء النظام من البداية إلى النهاية. يجمع هذا المشروع بين تقنيات من التعلم العميق ومعالجة الكلام وفهم اللغة الطبيعية القائم على القواعد لإنشاء أداة عملية ومفيدة.

  *الكلمات المفتاحية*: الاستماع والحضور والتهجئة (#en_std([LAS]))، التعرف على الكلام، المترجم القائم على القواعد، التفاعل بين الإنسان والحاسوب، إمكانية الوصول، أوامر لينكس، معدل الخطأ في الكلمات (#en_std([WER])).
])

#pagebreak()

#heading(
  en([Abstract]),
  level: 1,
  outlined: false,
  numbering: none,
)

#en_std([
  The command-line interface (CLI) is a powerful tool for interacting with computers, but it can be inaccessible for users with physical disabilities that make typing difficult or impossible. This project aims to solve this problem by creating a system that translates spoken language into executable Linux commands, thereby making the CLI more accessible and efficient for everyone.

  The primary goal is to develop a robust dual-system that accurately converts speech to commands. This is achieved by integrating two core components: a deep learning model for speech recognition and a deterministic interpreter for command generation. The speech recognition component is based on the "Listen, Attend, and Spell" (LAS) architecture, which is designed to transcribe spoken words into text. This text is then processed by a rule-based interpreter that maps natural language phrases to their corresponding syntactically correct Linux commands.

  The system's effectiveness is evaluated using two key metrics: Word Error Rate (WER) to measure the accuracy of the speech transcription model, and overall command accuracy to assess the performance of the end-to-end system. This project combines techniques from deep learning, speech processing, and rule-based natural language understanding to create a practical and beneficial tool.

  *Keywords*: Listen, Attend, and Spell (LAS), Speech Recognition, Rule-Based Interpreter, Human-Computer Interaction, Accessibility, Linux Commands, Word Error Rate (WER).
])

#pagebreak()
