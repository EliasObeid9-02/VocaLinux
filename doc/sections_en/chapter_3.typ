#import "../helpers.typ": flex_captions, en, en_std

= #en([Chapter 3: Practical Work])

#en_std([
  This chapter details the system's design and implementation, covering the architecture, model training, and integration of the core components.
])

== #en([Implementation Environment and Technology Stack])

#en_std([
  The project was developed entirely in the *Python* programming language, leveraging its extensive ecosystem for scientific computing and deep learning. The implementation relies on a set of robust, open-source libraries, summarized in the table below.

  #figure(
    table(
      columns: (1fr, 1fr, 4fr),
      align: (left+horizon, center+horizon, left+horizon),
      [*Category*], [*Library/Tool*], [*Purpose in Project*],

      [Deep Learning],
      [`TensorFlow` + `Keras`],
      [Used for building, training, and evaluating the core LAS model. Provided essential layers like `Bidirectional`, `LSTM`, and `Dense` for constructing the network architecture.],

      [Numerical and Audio],
      [`NumPy`],
      [The fundamental package for efficient numerical operations and manipulation of data arrays, used throughout the data pipeline.],

      [Numerical and Audio],
      [`SoundFile`],
      [Employed to read and write `flac` audio files, converting raw audio data into a numerical format suitable for processing.],

      [Evaluation],
      [`Jiwer`],
      [Used to measure the performance of the final model. Provided a standardized implementation for calculating the Word Error Rate WER and Character Error Rate CER.],

      [Visualization],
      [`Matplotlib`],
      [Utilized for plotting training and validation metrics e.g., loss, WER to visualize model performance and inform decisions during the experimental process.]
    ),
    kind: table,
    caption: flex_captions(
      [A summary of the core open-source libraries and frameworks used to implement the system.],
      [Project Technology Stack]
    )
  )
])

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
  + *Loading*: All audio files are loaded and standardized to a sample rate of $16,000$ Hz.
  + *STFT*: A Short-Time Fourier Transform (STFT) is applied to the signal using a Hann window, a Fast Fourier Transform (FFT) size of $1024$, and a hop length of $512$. This analyzes the audio's frequency content over time.
  + *Mel Scale Conversion*: The resulting spectrogram is mapped to the Mel scale using a filterbank of $100$ Mel bins.
  + *Logarithmic Scaling*: Finally, a logarithmic function is applied to the magnitudes of the Mel spectrogram. This compresses the dynamic range of the features, which stabilizes and accelerates model training.
])

==== #en([Text Processing and Vocabulary Curation])

#en_std([
  The corresponding text transcripts are processed to create numerical target sequences for the model's decoder.
  + *Vocabulary*: A fixed, character-level vocabulary was defined to ensure consistency. It consists of $32$ unique tokens: the $26$ lowercase English letters, an apostrophe (`'`), a space character, and four special tokens (`<pad>`, `<unk>`, `<sos>`, `<eos>`).
  + *Tokenization and Normalization*: All transcript text is converted to lowercase. Each transcript is then tokenized into a sequence of its constituent characters.
  + *Special Tokens*: To signal the start and end of a sequence for the decoder, the `<sos>` (start of sentence) and `<eos>` (end of sentence) tokens are prepended and appended to each tokenized sequence, respectively.
  + *Integer Mapping*: Each character in the final sequence is mapped to its unique integer ID from the predefined vocabulary. Any character not found in the vocabulary is mapped to the `<unk>` (unknown) token ID. The `<pad>` token is used to pad all sequences within a batch to a uniform length.
])

== #en([Model Architecture and Implementation])

#en_std([
  The core of this project is a custom implementation of the Listen, Attend, and Spell (LAS) architecture, whose theoretical basis was described in Chapter 2. The model was built using the TensorFlow and Keras frameworks. This section details the specific layers, parameters, and design choices made for our implementation.
])

#pagebreak()

=== #en([Listener (Encoder) Implementation])

#en_std([
  Following the pyramidal BLSTM structure, our Listener is designed to process the input log-Mel spectrogram and produce a compact, high-level feature representation. The layers are arranged as follows:
  + *Input BLSTM Layer*: The input is first processed by a standard Bidirectional LSTM (BLSTM) layer with $256$ units in each direction.
  + *Layer Normalization*: The output is normalized to stabilize training.
  + *Pyramidal BLSTM (pBLSTM) Stack*: The normalized output is fed into a stack of three pBLSTM layers. Each layer reduces the time resolution by a factor of two and uses $256$ LSTM units per direction.

  The final output of the Listener is a sequence of feature vectors that is $8$ times shorter in the temporal dimension than the original input spectrogram.
])

=== #en([Speller (Decoder) Implementation])

#en_std([
  Our Speller is an attention-based decoder designed to generate the final text transcription. Its implementation leverages several of the advanced concepts outlined in Chapter 2.
  + *Character Embedding Layer*: The integer ID of the previously generated character is converted into a dense vector of dimension $256$.
  + *Decoder LSTM Stack*: The character embedding is processed by a stack of two unidirectional LSTM layers, each with $512$ units. To combat overfitting, both `dropout` and `recurrent_dropout` are applied.
  + *Location-Aware Multi-Head Attention*: At each step, a context vector is calculated using a *Location-Aware Attention* mechanism with *4 heads*. This allows the model to focus on different parts of the audio representation simultaneously while preventing it from losing its place in the sequence. The internal dimension of the attention mechanism is $512$.
  + *Character Distribution Layer*: The output from the decoder LSTM is passed through a final `Dense` layer with a *Softmax* activation, producing a probability distribution over the character vocabulary.
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
  + *Initial Warm-up*: An initial warm-up period was used, during which the sampling probability was held constant. This allowed the model to stabilize before being exposed to its own, potentially noisy, predictions.
  + *Ramp-up and Stabilization Cycles*: Following the warm-up, the training proceeded in cycles. The sampling probability was gradually increased over a set number of epochs, followed by a stabilization period where the probability was held constant. This iterative process allowed the model to adapt to the increasing difficulty without becoming unstable.

  The probability, $epsilon$, is updated at the beginning of each epoch during a ramp phase. The new probability for the current epoch, $epsilon_e$, is calculated by adding a fixed linear increment to the probability from the previous epoch, $epsilon_(e-1)$.

  #pagebreak()

  This update rule can be expressed as:

  #math.equation(
    [
      $epsilon_e = epsilon_(e-1) + (epsilon_f - epsilon_s) / E_r$
    ],
    block: true
  )

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

  The following figures show the progression of the metrics over the $240$ epochs of training of Phase 1:

  #grid(
    columns: 2,
    gutter: 4pt,
    // Figure 1: Model Loss
    figure(
      image("../media/Training_Phase-1_Model-Loss.png", width: 100%),
      kind: image,
      caption: flex_captions(
        [Model loss over the $240$ epochs of Phase 1 training.],
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

      table.cell(rowspan: 2, [$15$]),
      [Sampled], [di we really know the mountain well when we are not acquainted with the cave n],
      [Non-Sampled], [```text 'aaa  wweeeeddiinneeeddiinneeeddii  wwooddeer        aann          asssssss  aann             aann        asssssss  aann             hhettiinnsss        asssssss  aann             aann          hheee             aann        hhettiinnsss        asssssss  aann             aann   '```],

      table.cell(rowspan: 2, [$120$]),
      [Sampled], [do we really know thatmountain well ween we are not accuainted tith aaa cavern],
      [Non-Sampled], [```text 'tee  rrr                                                                       '```],

      table.cell(rowspan: 2, [$240$]),
      [Sampled], [do ye rellly koow the mountain well whon we are not a quainted mith the cavern],
      [Non-Sampled], [```text 'doo   rrellly  ooww tthe  nnn tto wwell wwill  werre nntttt   qquiint  ff miicc  ccaareen'```],
    ),
    kind: table,
    caption: flex_captions(
      [A comparison of model outputs for the same audio sample at different epochs during Phase 1 training.],
      [Phase 1 Outputs Comparison]
    )
  )

  The results in the table clearly demonstrate steady improvements of non-sampled outputs over time. At first it was random incoherent output, but at the end word patterns have started emerging indicating the model is learning to starting to recognize words. On the other hand, sampled outputs degraded with the increased sampling probability but they are still coherent.
])

=== #en([Phase 2: Enhancing Attention with Location-Awareness])

#en_std([
  The results from Phase 1 established a solid baseline, but also highlighted a key challenge: the standard Multi-Head Attention struggled with longer audio sequences, sometimes losing its place or failing to transcribe the full utterance. The objective of Phase 2 was to address this by replacing the standard attention mechanism with a more robust *Location-Aware Attention* mechanism.

  This architectural change required a careful continuation of the training process. To allow the model to adapt, the weights of the new attention sub-layers were re-initialized. The model was then trained for an additional $140$ epochs.

  To help the model adjust to the new architecture, the scheduled sampling probability was reset to a lower value and then gradually increased over the training period. The learning rate and gradient clipping values were maintained from the end of the previous phase.

  #figure(
    table(
      columns: (2fr, 1fr, 1fr),
      align: (left, center, center),
      [*Hyperparameter*], [*Initial Value*], [*Final Value*],
      [Learning Rate], [$0.0003$], [$0.0003$],
      [Gradient Clipping Norm], [$2.5$], [$2.5$],
      [Sampling Probability (`epsilon`)], [$0.1$], [$0.5$],
    ),
    kind: table,
    caption: flex_captions(
      [Hyperparameter values during the $140$ epochs of Phase 2 training.],
      [Phase 2 Hyperparameters]
    )
  )

  The following figures show the progression of the metrics over the $140$ epochs of training in Phase 2.

  #grid(
    columns: 2,
    gutter: 4pt,
    // Figure 1: Model Loss
    figure(
      image("../media/Training_Phase-2_Model-Loss.png", width: 100%),
      kind: image,
      caption: flex_captions(
        [Model loss over the 140 epochs of Phase 2 training.],
        [Phase 2 Model Loss]
      )
    ),

    // Figure 2: Model Accuracy
    figure(
      image("../media/Training_Phase-2_Model-Accuracy.png", width: 100%),
      kind: image,
      caption: flex_captions(
        [Per-character accuracy during Phase 2 training.],
        [Phase 2 Model Accuracy]
      )
    ),

    // Figure 3: Character Error Rate (CER)
    figure(
      image("../media/Training_Phase-2_Model-CER.png", width: 100%),
      kind: image,
      caption: flex_captions(
        [Character Error Rate (CER) during Phase 2 training.],
        [Phase 2 Model CER]
      )
    ),

    // Figure 4: Word Error Rate (WER)
    figure(
      image("../media/Training_Phase-2_Model-WER.png", width: 100%),
      kind: image,
      caption: flex_captions(
        [Word Error Rate (WER) during Phase 2 training.],
        [Phase 2 Model WER]
      )
    ),
  )

  The quantitative success of this architectural change is also clear in the performance metrics. During this phase, both the CER and WER surpassed the best values from Phase 1, setting new records for the model.

  The following tables shows multiple sample transcriptions from different points during this phase, we also include outputs from the end of Phase 1 to illustrate the improvements gained from applying Location-Aware Attention.

  #figure(
    table(
      columns: (1fr, 3fr, 8fr),
      align: (center+horizon, center+horizon, center+horizon),
      inset: 8pt,
      [*Phase/Epoch*], [*Output Type*], [*Transcription*],

      [$1$/$240$], [Beam Search], [```text 'whnn it is a queestion of proving alooond aass  ooiitty ssnnss ween hassaaddbbeen considered floweed to go toooo tthrr tto go ttotthe bottom'```],
      [$2$/$20$], [Beam Search], [```text 'when iss accuushhddiin off rrogginng a ooon    gollf  a  so iide  since  wee  has aad bbeen considered faamm to go  to  ffrr tto ggo  to  hhe bottom'```],
      [$2$/$80$], [Beam Search], [```text 'when it is accuusseed ee  ff rroiing awwoodd   gollf aass  syy t  since hhen has it been considered froww tto go to ffor tto go ttotthe botuum'```],
      [$2$/$140$], [Beam Search], [```text 'when it is a question oof proping a ooond  ggolf aa  sosside    ssnnc  ween has i  been considered frowwntto go too far to go to the bathom'```],
    ),
    kind: table,
    caption: flex_captions(
      [Output comparison for the sentence: "when it is a question of probing a wound a gulf a society since when has it been considered wrong to go too far to go to the bottom".],
      [Medium Length Sample Comparison]
    )
  )

  From the comparison we notice that the model is more resistant to earlier errors and is now able to output correct words even after mistakes.

  #figure(
    table(
      columns: (1fr, 3fr, 8fr),
      align: (center+horizon, center+horizon, center+horizon),
      inset: 8pt,
      [*Phase/Epoch*], [*Output Type*], [*Transcription*],

      [1/240], [Beam Search], [```text 'to keep aee loot annd to resk you frr   llivinn toowwhole ttee goll whhereiit by  ffragmments of somme llnnguunn wwhich man has spoken  nn  which wooeedoott ofwwhich ttss compoccated too eexteend  oftthe records oofssocial observatioon issseeltth'```],
      [2/20], [Beam Search], [```text 'to  eee  lott annd too uus  you  ff   bbliivinn  oo  hhole abbove thhe goll  wheeriit buut iffrrggmeents oof some llnnguage wwhichmman has spokken oof wwhich  siveaalizaationnaas commosee  orrr  vve  whhich it iss complitatee  twweettsstnnd  of tthe recorrs  of sss'```],
      [2/80], [Beam Search], [```text 'to keep a foo  and too rest you foo  a bivvi  in  nn  ooll tto oove hhe golf  whereiit but iffraagmennt  o  some language which man has spoken and whicch would other iise be lost that  is oo say one of the elements good or aad off whicch iivilizaatiooniis composed '```],
      [2/140], [Beam Search], [```text 'to people fol  and to rescue froom a blivioo  nn  hhole to  oov  the golf wher  it but ifraagment oofssome language which man has spoken andwwhich woudd otherwiiee beloost mataas  oo say oone of tthe elements good or aad of whicc  sivilizatiooniis composed oor  by '```],
    ),
    kind: table,
    caption: flex_captions(
      [Output comparison for the sentence: "to keep afloat and to rescue from oblivion to hold above the gulf were it but a fragment of some language which man has spoken and which would otherwise be lost that is to say one of the elements good or bad of which civilization is composed or by which it is complicated to extend the records of social observation is to serve civilization itself".],
      [Long Length Sample Comparison]
    )
  )

  Again, this long length sample demonstrates the improvements the model gained from location-awareness, for example the transcript from phase 1 has missed a lot of word patterns such as the one for "above" and "civilization" while at the end of phase 2 the model generated word patterns for them.
])

#pagebreak()

=== #en([Phase 3: Combating Overfitting with Regularization])

#en_std([
  Following the architectural improvements in Phase $2$, a clear pattern of overfitting began to emerge. While the training metrics continued to improve, the validation metrics started to plateau, indicating that the model was beginning to memorize the training data rather than learning to generalize. Although the validation metrics showed signs of plateauing throughout training, the cause was attributed to low sampling rate.

  The objective of this brief, $55$-epoch phase was to combat this overfitting by introducing stronger regularization. The primary strategy was to increase the dropout rates in the decoder and apply dropout to the encoder for the first time.

  To increase regularization, the `dropout` and `recurrent_dropout` values in the Speller's LSTM stack were raised. Additionally, a new `dropout` layer was added to the Listener's pBLSTM stack to regularize the encoder. Other hyperparameters were maintained from the end of the previous phase.

  #figure(
    table(
      columns: (2fr, 1fr, 1fr),
      align: (left, center, center),
      [*Hyperparameter*], [*Initial Value*], [*Final Value*],
      [Learning Rate], [$0.0003$], [$0.0003$],
      [Gradient Clipping Norm], [$2.5$], [$2.5$],
      [Sampling Probability (`epsilon`)], [$0.5$], [$0.5$],
      [Decoder Dropout], [$0.17$], [$0.3$],
      [Encoder Dropout], [$0.0$], [$0.1$],
    ),
    kind: table,
    caption: flex_captions(
      [Hyperparameter values during the 55 epochs of Phase 3 training.],
      [Phase 3 Hyperparameters]
    )
  )

  The model was trained for 55 epochs with the enhanced regularization. The following figures show the progression of the metrics during this phase.

  #grid(
    columns: 2,
    gutter: 4pt,
    // Figure 1: Model Loss
    figure(
      image("../media/Training_Phase-3_Model-Loss.png", width: 100%),
      kind: image,
      caption: flex_captions(
        [Model loss over the 55 epochs of Phase 3 training.],
        [Phase 3 Model Loss]
      )
    ),

    // Figure 2: Model Accuracy
    figure(
      image("../media/Training_Phase-3_Model-Accuracy.png", width: 100%),
      kind: image,
      caption: flex_captions(
        [Per-character accuracy during Phase 3 training.],
        [Phase 3 Model Accuracy]
      )
    ),

    // Figure 3: Character Error Rate (CER)
    figure(
      image("../media/Training_Phase-3_Model-CER.png", width: 100%),
      kind: image,
      caption: flex_captions(
        [Character Error Rate (CER) during Phase 3 training.],
        [Phase 3 Model CER]
      )
    ),

    // Figure 4: Word Error Rate (WER)
    figure(
      image("../media/Training_Phase-3_Model-WER.png", width: 100%),
      kind: image,
      caption: flex_captions(
        [Word Error Rate (WER) during Phase 3 training.],
        [Phase 3 Model WER]
      )
    ),
  )

  The analysis of this phase was conclusive. While the training metrics continued to show stabilization or slight improvement, the validation metrics showed no clear signs of progress. The validation WER and CER failed to improve upon the records set at the end of Phase $2$. Because the increased regularization did not yield the desired improvement in generalization, this experimental phase was concluded. This outcome prompted a shift in strategy towards exploring different potential solutions.

  Many attempts were tried in order to combat the overfitting problem. By thorough review of the original research paper, the model code, and the training strategy employed, we determined one of the causes of the problem to be the discrepancy between the size of training data and the network size. For comparison, the research paper mentioned that they trained the model on $2000$ hours of training using the same network size employed in our implementation, while our implementation was trained on only $100$ hours of training data.

  This large discrepancy prompted us to consider reducing the network size massively, applying heavy augmentation through the SpecAugment method, and using a larger dataset such as the $360$ hour split. Another technique tried was *Cyclical Learning Rate Schedule*. The model versions using these modifications where trained for a variety of epochs ranging from $20$ all the way to $100$, and most versions didn't provide any improvements or potential signs of solving the overfitting problem.
])

== #en([The Rule-Based Interpreter])

#en_std([
  The output from the ASR model is a raw text transcription, which, while often accurate, is not a syntactically valid shell command. Spoken language is inherently ambiguous and contains phonetic spellings (e.g., "slash," "dash") that a terminal cannot execute. The final component of this project is a rule-based interpreter designed to bridge this gap. Its sole purpose is to parse the raw ASR output and translate it into a well-formed, executable Linux command.

  The interpretation process is handled in two main stages: first, a *Translator* converts spoken keywords into their corresponding characters and commands, and second, a *Path Resolver* intelligently corrects potential spelling errors in file and directory paths.
])

=== #en([Translation and Vocabulary Mapping])

#en_std([
  The first stage of interpretation is managed by the `Translator` class. It operates on a predefined vocabulary of keywords that map spoken words to their intended text representation. This vocabulary is organized into several categories.o
])

==== #en([Recognized Vocabulary])

#en_std([
  The interpreter's vocabulary is explicitly defined to handle commands, special characters, digits, and a phonetic alphabet for spelling.

  - *Commands*: A core set of common Linux commands are recognized.
  - *Special Characters*: Spoken words like "slash," "dot," or "double quote" are mapped to their symbolic equivalents (`/`, `.`, `"`).
  - *Digits*: Spoken numbers ("zero" through "nine") are converted to their digit form.
  - *Phonetic Alphabet*: A standard phonetic alphabet (e.g., "adam" for "a", "boy" for "b") is used to allow for precise, unambiguous spelling of filenames or arguments.
d
  The table below details the primary keywords recognized by the interpreter.

  #figure(
    table(
      columns: (2fr, 4fr),
      align: (left, left),
      [*Category*], [*Examples*],
      [Commands], [`ls`, `cd`, `cp`, `mv`, `rm`, `mkdir`, `echo`, `touch`],
      [Special Characters], [`slash` -> `/`, `hyphen` -> `-`, `double quote` -> `"`],
      [Digits], [`one` -> `1`, `five` -> `5`, `nine` -> `9`],
      [Phonetic Alphabet], [`adam` -> `a`, `robert` -> `r`, `zebra` -> `z`],
    ),
    kind: table,
    caption: flex_captions(
      [Categories and examples from the interpreter's recognized vocabulary.],
      [Interpreter Vocabulary]
    )
  )
])

==== #en([Multi-Word Keyword Detection])

#en_std([
  To handle keywords that consist of more than one word (e.g., "double quote"), the translator implements a longest-match-first algorithm. It scans the input text for the longest possible keyword sequence at any given position. For example, when processing the phrase "double quote my file," it will first identify "double quote" as a single, two-word token and translate it to `"`, rather than processing "double" and "quote" as separate, unknown words.
])

==== #en([Keywords Escaping Mechanism])

#en_std([
  To handle cases where a keyword is intended as a literal argument, the system includes an escape mechanism. The keyword *backslash* preceding any other recognized keyword causes the interpreter to treat the subsequent keyword as plain text, ignoring its special function. For example, the phrase "echo space backslash space" would result in the command `echo space`, correctly treating "space" as a literal word rather than an empty space between words.
])

=== #en([Path Resolution and Correction])

#en_std([
  A common source of error in speech recognition is the misspelling of filenames or directory names. The `PathResolver` class is designed to mitigate this issue by intelligently correcting paths.

  After the initial translation, the interpreter splits the command into parts. Any part containing a path separator (`/`) is passed to the path resolver. The resolver then processes the path segment by segment. For each segment, it checks the corresponding directory on the filesystem and uses a fuzzy string matching algorithm (`difflib.get_close_matches`) to find the closest matching file or directory name. If a close match (with a similarity cutoff of $0.5$) is found, it replaces the potentially misspelled segment with the correct name.

  This mechanism works for both absolute and relative paths, significantly increasing the robustness of the system by correcting minor transcription errors that would otherwise cause the final command to fail.
])
