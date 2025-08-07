#import "../helpers.typ": en, en_std

= #en([Chapter 1: Introduction])

#en_std([
  The Linux command-line interface (CLI) stands as a paragon of power and efficiency in the world of computing. For developers, system administrators, and power users, it is the quintessential environment for precise system control, offering unparalleled speed for complex tasks through succinct text-based commands. This power, however, is predicated on a significant assumption: the user's ability to interact with a keyboard rapidly and accurately. This fundamental reliance on manual typing creates a formidable barrier, transforming this tool of efficiency into a source of exclusion for many skilled individuals.
])

== #en([The Digital Barrier of the Command Line])

#en_std([
  At the heart of this project are the users for whom the standard keyboard is not an instrument of productivity, but an obstacle. This includes individuals with a range of motor impairments, such as Repetitive Strain Injury (RSI), arthritis, and paralysis, which can make prolonged or intricate typing painful, slow, or altogether impossible.

  While modern Graphical User Interfaces (GUIs) have made significant strides in accessibility, incorporating features like screen readers, high-contrast modes, and robust voice control, the command-line environment has largely remained a spartan landscape. A critical accessibility gap has emerged: the very tool that offers the deepest level of system control is the least accommodating to those who cannot use a keyboard. This project directly confronts this gap, seeking to bridge the divide between human intent and command-line execution through voice.
])

== #en([A Voice-Enabled Gateway to Linux])

#en_std([
  To address the identified accessibility gap, this project proposes a dual-component system designed to act as a voice-enabled gateway to the Linux shell. The proposed solution is architected to be both effective and lightweight, ensuring its usability across a wide spectrum of hardware. This requires not only a sophisticated interpreter but also a highly specialized speech recognition component, fine-tuned for the unique syntax of the command line.

  #pagebreak()

  The system comprises two core modules working in concert:
  + *A Modified Speech Recognition (ASR) System*: This module's responsibility is to capture the user's spoken words and transcribe them into raw text. It must be tailored to recognize the specific vocabulary and phonetic patterns of shell commands, including the ability to discern individual characters for spelling out arguments.
  + *A Specialized Command Interpreter*: This is the logical core of the project. It receives the raw text from the ASR system and performs several crucial tasks: it maps the unstructured text to a valid command, checks for syntactical errors, corrects common spelling mistakes, and intelligently constructs valid file paths from the user's speech.

  A key design principle is resource efficiency. For this reason, the use of Large Language Models (LLMs) was explicitly ruled out. While powerful, LLMs carry substantial computational overhead and hardware requirements that conflict with the goal of creating a broadly accessible and deployable tool.
])

== #en([Aims and Objectives])

#en_std([
  The primary aim of this project is to design, implement, and evaluate a prototype system that provides a hands-free, voice-based interface for the Linux shell, with a focus on improving accessibility for users with physical disabilities.

  To achieve this aim, the following objectives have been established:
  + To identify a suitable Automatic Speech Recognition (ASR) model that is lightweight, powerful, and capable of independent character recognition.
  + To train and modify the selected ASR model to improve its performance and accuracy specifically for transcribing Linux shell commands.
  + To design a system architecture that effectively integrates the modified ASR module with a custom-built command interpretation engine.
  + To develop a working prototype that can recognize and correctly formulate a core set of common Linux commands.
  + To evaluate the prototype's accuracy and usability based on a predefined set of spoken command scenarios.
])

== #en([Scope and Limitations])

#en_std([
  To ensure the project is achievable within the given timeframe, its scope has been carefully defined. The project will focus exclusively on the `bash` shell, the English language, and a curated set of commands from the GNU `coreutils` package.

  *Commands in Scope*:
  #align(center+top, grid(
      rows: (1.5em, 1.5em, 1.5em),
      columns: (1fr, 1fr, 1fr),
      `ls`,    `cd`,   `cp`,
      `mv`,    `rm`,   `mkdir`,
      `rmdir`, `echo`, `touch`,
    )
  )

  While this project will not create a speech-recognition engine from scratch, a significant component of the work involves the selection, modification, and fine-tuning of an existing ASR model to meet the specific demands of command-line transcription. The system will not, in its initial version, support advanced shell features such as command chaining (piping), redirection, or the execution of shell scripts. These areas are noted as potential avenues for future work.
])

== #en([Referenced Research Papers])

#en_std([
  This project builds upon the foundations laid by several key research papers in the field of speech recognition and deep learning. The following papers provide the theoretical and practical basis for the core components of our system.
])

=== #en([Listen, Attend, and Spell])

#en_std([
  This paper introduces the Listen, Attend, and Spell (LAS) model, an end-to-end neural network for automatic speech recognition. It proposes a novel architecture consisting of a "Listener" (an encoder) that processes the audio input and a "Speller" (a decoder) that uses an attention mechanism to generate the text transcription. The LAS model simplifies the traditional speech recognition pipeline by learning to transcribe speech directly from audio, which is the core of our speech-to-text component.
])

#pagebreak()

=== #en([Attention is All You Need])

#en_std([
  This seminal paper introduces the Transformer architecture, which dispenses with recurrence and convolutions entirely, relying solely on attention mechanisms. While the full Transformer model is not used in this project, we draw from its key innovation: Multi-Head Attention. This mechanism enhances standard attention by running it multiple times in parallel. Each of these "attention heads" learns to focus on different parts of the input, allowing the model to jointly attend to information from different representation subspaces simultaneously. This concept is leveraged to improve the representational power of our model's attention layer.
])

=== #en([Attention-Based Models for Speech Recognition])

#en_std([
  This paper further explores the use of attention mechanisms in speech recognition. It introduces a location-aware attention mechanism that helps the model to better handle long utterances and repetitions. This is particularly relevant for our project, as it allows the model to maintain focus on the relevant parts of the audio signal when transcribing longer and more complex commands.
])

=== #en([SpecAugment: A Simple Data Augmentation Method])

#en_std([
  SpecAugment presents a simple yet highly effective data augmentation technique for speech recognition. It operates directly on the log-mel spectrogram of the audio input, applying time warping, frequency masking, and time masking to create more robust models. By incorporating SpecAugment into our training pipeline, we aim to improve the model's resilience to variations in speech and background noise.
])

=== #en([Scheduled Sampling for Sequence Prediction])

#en_std([
  This paper addresses a common issue in training sequence prediction models: the discrepancy between training (using "teacher forcing") and inference (using the model's own predictions). Scheduled Sampling provides a curriculum learning strategy that gradually shifts from teacher forcing to using the model's own predictions during training. This helps to close the gap between training and inference, leading to better performance in real-world scenarios.
])

=== #en([Cyclical Learning Rates for Training Neural Networks])

#en_std([
  This paper challenges the conventional wisdom of monotonically decreasing the learning rate. It proposes Cyclical Learning Rates (CLR), a method where the learning rate cyclically varies between a minimum and maximum bound. This technique can lead to faster training and improved performance by helping the model traverse saddle points in the loss landscape more effectively. This concept forms the basis for an advanced optimization strategy explored in the later stages of our model training.
])

== #en([Documentation Structure])

#en_std([
  This document is structured to provide a comprehensive overview of the project. Chapter 2 delves into the theoretical foundations of the technologies used, including the LAS model. Chapter 3 details the system's design and implementation, covering the architecture, model training, and integration of the components.
])

#pagebreak()
