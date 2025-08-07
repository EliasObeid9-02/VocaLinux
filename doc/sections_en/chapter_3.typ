#import "../helpers.typ": flex_captions, en, en_std

= #en([Chapter 3: Practical Work])

#en_std([
  This chapter details the system's design and implementation, covering the architecture, model training, and integration of the core components. The first component we will discuss is the rule-based interpreter, which translates the output of the speech recognition model into executable commands.
])

== #en([Implementation Environment and Technology Stack])

#en_std([
  The project was developed entirely in the *Python* programming language, leveraging its extensive ecosystem for scientific computing and deep learning. The implementation relies on a set of robust, open-source libraries, summarized in the table below.
])

#figure(
  table(
    columns: (1fr, 1fr, 4fr),
    align: (left+horizon, center+horizon, left+horizon),
    en_std([*Category*]), en_std([*Library/Tool*]), en_std([*Purpose in Project*]),

    en_std([Deep Learning]),
    en_std([TensorFlow and Keras]),
    en_std([Used for building, training, and evaluating the core LAS model. Provided essential layers like `Bidirectional`, `LSTM`, and `Dense` for constructing the network architecture.]),

    en_std([Numerical and Audio]),
    en_std([NumPy]),
    en_std([The fundamental package for efficient numerical operations and manipulation of data arrays, used throughout the data pipeline.]),

    en_std([Numerical and Audio]),
    en_std([SoundFile]),
    en_std([Employed to read and write `.flac` audio files, converting raw audio data into a numerical format suitable for processing.]),

    en_std([Evaluation]),
    en_std([Jiwer]),
    en_std([Used to measure the performance of the final model. Provided a standardized implementation for calculating the Word Error Rate (WER) and Character Error Rate (CER).]),

    en_std([Visualization]),
    en_std([Matplotlib]),
    en_std([Utilized for plotting training and validation metrics (e.g., loss, WER) to visualize model performance and inform decisions during the experimental process.])
  ),
  kind: table,
  caption: flex_captions(
    en_std([A summary of the core open-source libraries and frameworks used to implement the system.]),
    en_std([Project Technology Stack])
  )
)

== #en([Dataset Implementation and Vocabulary Curation])

#en_std([
  The foundation of the model is built upon the *LibriSpeech* corpus. This section details the specific subsets used, the partitioning strategy employed, and the comprehensive preprocessing pipeline developed to transform the raw data into a format suitable for training the neural network.
])

=== #en([Data Selection and Partitioning])

#en_std([
  For this project, a focused subset of the LibriSpeech corpus was selected to balance training time with model performance.
  - *Training and Validation Sets*: The `train-clean-100` split, containing 100 hours of high-quality speech, was used as the primary source for training data. To effectively monitor for overfitting and guide the training process, this set was further partitioned: 80% of the data was allocated for model training, while the remaining $20%$ was reserved as a validation set.
  - *Test Set*: To provide an unbiased final evaluation of the model's performance on unseen data, the official `test-clean` split was used.

  During training, the data is loaded in batches of $32$ samples. The training set is shuffled at the beginning of each epoch to ensure the model does not learn the order of the examples, promoting better generalization.
])

=== #en([Preprocessing Pipeline])

#en_std([
  A two-stage preprocessing pipeline was implemented to handle the audio and text data respectively.
])

==== #en([Audio Processing: Log-Mel Spectrograms])

#en_std([
  The raw audio waveforms are transformed into *log-Mel spectrograms*, a feature representation that mimics human auditory perception and is highly effective for speech recognition tasks. This conversion follows a precise sequence of steps:
  
  #enum(
    [*Loading:* All audio files are loaded and standardized to a sample rate of $16,000$ Hz.],
    [*STFT:* A Short-Time Fourier Transform (STFT) is applied to the signal using a Hann window, a Fast Fourier Transform (FFT) size of $1024$, and a hop length of $512$. This analyzes the audio's frequency content over time.],
    [*Mel Scale Conversion:* The resulting spectrogram is mapped to the Mel scale using a filterbank of $100$ Mel bins.],
    [*Logarithmic Scaling:* Finally, a logarithmic function is applied to the magnitudes of the Mel spectrogram. This compresses the dynamic range of the features, which stabilizes and accelerates model training.]
  )
])

==== #en([Text Processing and Vocabulary Curation])

#en_std([
  The corresponding text transcripts are processed to create numerical target sequences for the model's decoder.

  #enum(
    [*Vocabulary*: A fixed, character-level vocabulary was defined to ensure consistency. It consists of $32$ unique tokens: the $26$ lowercase English letters, an apostrophe (`'`), a space character, and four special tokens (`<pad>`, `<unk>`, `<sos>`, `<eos>`).],
    [*Tokenization and Normalization*: All transcript text is converted to lowercase. Each transcript is then tokenized into a sequence of its constituent characters.],
    [*Special Tokens*: To signal the start and end of a sequence for the decoder, the `<sos>` (start of sentence) and `<eos>` (end of sentence) tokens are prepended and appended to each tokenized sequence, respectively.],
    [*Integer Mapping*: Each character in the final sequence is mapped to its unique integer ID from the predefined vocabulary. Any character not found in the vocabulary is mapped to the `<unk>` (unknown) token ID. The `<pad>` token is used to pad all sequences within a batch to a uniform length.]
  )
])

== #en([Model Architecture and Implementation])

#en_std([
  The core of this project is a custom implementation of the Listen, Attend, and Spell (LAS) architecture, whose theoretical basis was described in Chapter 2. The model was built using the TensorFlow and Keras frameworks. This section details the specific layers, parameters, and design choices made for our implementation.
])

#pagebreak()

=== #en([Listener (Encoder) Implementation])

#en_std([
  Following the pyramidal BLSTM structure, our Listener is designed to process the input log-Mel spectrogram and produce a compact, high-level feature representation. The layers are arranged as follows:

  #enum(
    [*Input BLSTM Layer*: The input is first processed by a standard Bidirectional LSTM (BLSTM) layer with $256$ units in each direction.],
    [*Layer Normalization*: The output is normalized to stabilize training.],
    [*Pyramidal BLSTM (pBLSTM) Stack*: The normalized output is fed into a stack of three pBLSTM layers. Each layer reduces the time resolution by a factor of two and uses $256$ LSTM units per direction.]
  )

  The final output of the Listener is a sequence of feature vectors that is $8$ times shorter in the temporal dimension than the original input spectrogram.
])

=== #en([Speller (Decoder) Implementation])

#en_std([
  Our Speller is an attention-based decoder designed to generate the final text transcription. Its implementation leverages several of the advanced concepts outlined in Chapter 2.

  #enum(
    [*Character Embedding Layer*: The integer ID of the previously generated character is converted into a dense vector of dimension $256$.],
    [*Decoder LSTM Stack*: The character embedding is processed by a stack of two unidirectional LSTM layers, each with $512$ units. To combat overfitting, both `dropout` and `recurrent_dropout` are applied.],
    [*Location-Aware Multi-Head Attention*: At each step, a context vector is calculated using a *Location-Aware Attention* mechanism with *4 heads*. This allows the model to focus on different parts of the audio representation simultaneously while preventing it from losing its place in the sequence. The internal dimension of the attention mechanism is $512$.],
    [*Character Distribution Layer*: The output from the decoder LSTM is passed through a final `Dense` layer with a *Softmax* activation, producing a probability distribution over the character vocabulary.],
  )
])

== #en([Experiments and Results])

#en_std([
  This section details the experimental process used to train and evaluate the model. It outlines the general training configuration before presenting the specific setup and results for each experimental phase.
])

=== #en([Training Configuration])

#en_std([
  Building on the principles outlined in Chapter 2, a consistent set of tools was employed:
  - *Optimizer*: The model was trained using the *Adam* optimizer. *Gradient clipping* was also applied to prevent exploding gradients.
  - *Loss Function*: A custom *Sparse Categorical Cross-Entropy* function was implemented to work with the model's softmax outputs and to automatically mask `<pad>` tokens.
  - *Evaluation Metrics*: Model performance was monitored using the training loss, per-character accuracy, Character Error Rate (CER), and Word Error Rate (WER).
])

=== #en([Scheduled Sampling Strategy])

#en_std([
  Building upon the standard scheduled sampling strategies discussed in Chapter 2, this project implements a custom strategy featuring a modified linear ramp-up and stabilization cycles. This approach was designed to bridge the gap between training and inference while keeping the model as stable as possible. This strategy's stages are:

  #enum(
    [*Initial Warm-up*: An initial warm-up period was used, during which the sampling probability was held constant. This allowed the model to stabilize before being exposed to its own, potentially noisy, predictions.],
    [*Ramp-up and Stabilization Cycles*: Following the warm-up, the training proceeded in cycles. The sampling probability was gradually increased over a set number of epochs, followed by a stabilization period where the probability was held constant. This iterative process allowed the model to adapt to the increasing difficulty without becoming unstable.]
  )

  The probability, $epsilon$, is updated at the beginning of each epoch during a ramp phase. The new probability for the current epoch, $epsilon_e$, is calculated by adding a fixed linear increment to the probability from the previous epoch, $epsilon_(e-1)$.

  #pagebreak()

  This update rule can be expressed as:

  #align(center)[
    $epsilon_e = epsilon_(e-1) + (epsilon_f - epsilon_s) / E_r$
  ]

  where:
  - $epsilon_e$: The sampling probability for the current epoch $e$.
  - $epsilon_(e-1)$: The sampling probability from the previous epoch.
  - $epsilon_s$: The starting probability at the beginning of the ramp phase.
  - $epsilon_f$: The final target probability at the end of the ramp phase.
  - $E_r$: The total number of epochs over which the probability is ramped.

  The resulting value is then clamped to ensure it does not overshoot the final target probability, $epsilon_f$. It also correctly handles the case where $epsilon_f < epsilon_s$ and decreases the probability without causing errors.
])

=== #en([Phase 1: Initial Training])

#en_std([
  The first phase of training, conducted over $240$ epochs, focused on establishing a baseline performance. During this phase, the architecture was configured with a standard *Multi-Head Attention* mechanism, without the location-aware component. The primary goal was to ensure the model could learn the fundamental task of transcribing speech and to find a stable set of hyperparameters.

  The table below details the key hyperparameters used during this phase.

  #figure(
    table(
      columns: (2fr, 1fr, 1fr),
      align: (left, center, center),
      [*Hyperparameter*], [*Initial Value*], [*Final Value*],
      [Learning Rate], [$0.001$], [$0.0003$],
      [Gradient Clipping Norm], [$5.0$], [$2.5$],
      [Sampling Probability (`epsilon`)], [$0.05$], [$0.5$],
    ),
    kind: table,
    caption: flex_captions(
      [Learning Rate, Gradient Clipping Norm, and Sampling probability values during Phase 1 of training.],
      [Phase 1 Hyperparameters]
    )
  )

  #pagebreak()

  The following figures show the progression of the metrics over the 240 epochs of training of Phase 1:

  #grid(
    columns: 2,
    gutter: 4pt,
    // Figure 1: Model Loss
    figure(
      image("../media/Training_Phase-1_Model-Loss.png", width: 100%),
      kind: image,
      caption: flex_captions(
        [Model loss over the 240 epochs of Phase 1 training.],
        [Phase 1 Model Loss]
      )
    ),

    // Figure 2: Model Accuracy
    figure(
      image("../media/Training_Phase-1_Model-Accuracy.png", width: 100%),
      kind: image,
      caption: flex_captions(
        [Per-character accuracy during Phase 1 training.],
        [Phase 1 Model Accuracy]
      )
    ),

    // Figure 3: Character Error Rate (CER)
    figure(
      image("../media/Training_Phase-1_Model-CER.png", width: 100%),
      kind: image,
      caption: flex_captions(
        [Character Error Rate (CER) during Phase 1 training.],
        [Phase 1 Model CER]
      )
    ),

    // Figure 4: Word Error Rate (WER)
    figure(
      image("../media/Training_Phase-1_Model-WER.png", width: 100%),
      kind: image,
      caption: flex_captions(
        [Word Error Rate (WER) during Phase 1 training.],
        [Phase 1 Model WER]
      )
    ),
  )

  #pagebreak()

  The following table shows a sample transcriptions from different points during this phase, illustrating the model's learning progress. The sample transcription is "do we really know the mountain well when we are not acquainted with the cavern".

  #figure(
    table(
      columns: (1fr, 3fr, 8fr),
      align: (center+horizon, center+horizon, center+horizon),
      inset: 8pt,
      [*Epoch*], [*Output Type*], [*Transcription*],

      table.cell(rowspan: 2, [15]),
      [Sampled Output], [di we really know the mountain well when we are not acquainted with the cave n],
      [Non-Sampled Output], [```text 'aaa  wweeeeddiinneeeddiinneeeddii  wwooddeer        aann          asssssss  aann             aann        asssssss  aann             hhettiinnsss        asssssss  aann             aann          hheee             aann        hhettiinnsss        asssssss  aann             aann   '```],

      table.cell(rowspan: 2, [120]),
      [Sampled Output], [do we really know thatmountain well ween we are not accuainted tith aaa cavern],
      [Non-Sampled Output], [```text 'tee  rrr                                                                       '```],

      table.cell(rowspan: 2, [240]),
      [Sampled Output], [do ye rellly koow the mountain well whon we are not a quainted mith the cavern],
      [Non-Sampled Output], [```text 'doo   rrellly  ooww tthe  nnn tto wwell wwill  werre nntttt   qquiint  ff miicc  ccaareen'```],
    ),
    kind: table,
    caption: flex_captions(
      [A comparison of model outputs for the same audio sample at different epochs during Phase 1 training.],
      [Phase 1 Outputs Comparison]
    )
  )

  The results in the table clearly demonstrate steady improvements of non-sampled outputs over time. At first it was random incoherent output, but at the end word patterns have started emerging indicating the model is learning to starting to recognize words. On the other hand, sampled outputs degraded with the increased sampling probability but they are still coherent.
])

// TODO: complete this section
== #en([The Rule-Based Interpreter])

// #en_std([
// ])
//
// === Escaping Keywords
//
// To handle cases where a keyword (like "list" or "slash") is intended as a literal argument rather than a command or special character, the system includes an escape mechanism. The keyword "backslash" preceding any other recognized keyword causes the interpreter to treat the subsequent keyword as plain text, ignoring its special function. For example, the phrase "copy backslash list to home" would result in the command `cp list /home`, treating "list" as a literal filename.
//
// === Recognized Command Vocabulary
//
// The interpreter is designed to recognize a specific set of spoken keywords and map them to a predefined vocabulary of Linux commands. The following table lists a representative subset of the supported commands, their corresponding spoken keywords, and a brief description of their function.
//
// #figure(
//   table(
//     columns: (1fr, 1fr, 2fr),
//     align: (center+horizon, center+horizon, center+horizon),
//     [*Spoken Keyword(s)*], [*Generated Command*], [*Description*],
//     ["list"], [`ls`], [Lists directory contents.],
//     ["change directory"], [`cd`], [Changes the current directory.],
//     ["copy"], [`cp`], [Copies files or directories.],
//     ["move"], [`mv`], [Moves or renames files or directories.],
//     ["remove"], [`rm`], [Removes files or directories.],
//     ["make directory"], [`mkdir`], [Creates a new directory.],
//     ["echo"], [`echo`], [Displays a line of text.],
//     ["touch"], [`touch`], [Creates an empty file.],
//   ),
//   kind: table,
//   caption: flex_captions(
//     [List of commands recognized by the interpreter.],
//     [Recognized Commands]
//   )
// )
//
