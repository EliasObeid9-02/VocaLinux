#import "../helpers.typ": en, en_std

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
