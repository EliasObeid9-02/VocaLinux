#import "../helpers.typ": flex_captions, en, en_std

= #en([Chapter 2: Theoretical Foundations])

#en_std([
  Automatic Speech Recognition (ASR), the technology that enables machines to understand and transcribe human speech, has evolved from a niche scientific pursuit into a ubiquitous feature of modern life. From virtual assistants on smartphones to voice-controlled home devices, ASR systems have fundamentally changed how we interact with technology. At its core, the goal of any ASR system is to solve a complex pattern recognition problem: converting a continuous, variable audio waveform into a discrete, structured sequence of text.

  This chapter delves into the theoretical underpinnings of the technologies required to build a modern ASR system. We will begin by tracing the evolution of ASR architectures to provide context for the current state of the art. Following this, we will explore the specific components and strategies that form the foundation of this project, from the dataset and model architecture to the principles of training and evaluation.
])

== #en([The Evolution of Speech Recognition Architectures])

#en_std([
  The field of ASR has undergone a significant paradigm shift over the last two decades. The journey from complex, multi-stage pipelines to unified, end-to-end neural networks marks a critical evolution in the pursuit of more accurate and efficient speech transcription.
])

=== #en([Traditional Hybrid Models])

#en_std([
  For many years, the state of the art in ASR was dominated by hybrid systems that combined several independently trained components. These traditional pipelines typically included:
  - An *Acoustic Model*, often a Gaussian Mixture Model (GMM) paired with a Hidden Markov Model (HMM), responsible for mapping short audio frames to phonetic units.
  - A *Pronunciation Model* or lexicon, which provided a mapping from phonetic units to words.
  - A *Language Model*, usually an n-gram model, which estimated the probability of a sequence of words, adding linguistic context to the predictions.

  While powerful for their time, these systems were complex to build and maintain. Each component had to be carefully engineered and trained on different data, and the hard-coded dependencies between them made the entire pipeline brittle and difficult to optimize as a single unit.
])

=== #en([The Rise of End-to-End Architectures])

#en_std([
  The advent of deep learning brought about a revolution in ASR with the development of *end-to-end* models. These systems replace the entire complex pipeline of traditional models with a single, unified neural network that learns to transcribe speech directly from audio input. This approach offers several key advantages:
  - *Simplicity*: The need for hand-engineered components like pronunciation models is eliminated.
  - *Joint Optimization*: The entire system can be trained jointly on pairs of audio and text, allowing the model to learn a direct mapping and optimize all its internal parameters for the single goal of accurate transcription.
  - *Performance*: End-to-end models have consistently outperformed traditional hybrid systems, setting new standards for accuracy.

  Two primary architectures have emerged as leaders in the end-to-end space:
  + *Connectionist Temporal Classification (CTC)*: CTC-based models use a specialized loss function that allows the network to handle the alignment between variable-length audio frames and shorter text sequences without needing a separate attention mechanism.
  + *Attention-Based Sequence-to-Sequence Models*: These models, such as the *Listen, Attend, and Spell (LAS)* architecture at the core of this project, use an attention mechanism to explicitly learn the alignment between the audio input and the output text at each step.

  The decision to focus on the LAS architecture for this project was driven by its elegant design and its proven effectiveness in producing highly accurate, character-level transcriptions, making it a powerful foundation for building a domain-specific speech recognition system.
])

#pagebreak()

== #en([Foundational Datasets: The LibriSpeech Corpus])

#en_std([
  A cornerstone in modern Automatic Speech Recognition (ASR) research is the *LibriSpeech* corpus. First introduced by Vassil Panayotov et al., it is a large-scale dataset containing approximately 1000 hours of read English speech, derived from the audiobooks of the LibriVox project. Its substantial size, clean audio recordings, and well-defined partitions have established it as a standard benchmark for developing and evaluating ASR systems.
])

=== #en([Corpus Structure and Organization])

#en_std([
  The LibriSpeech corpus is organized in a hierarchical filesystem structure that is logical and easy to parse. The data is sorted by speaker ID and chapter ID, ensuring that all utterances from a single chapter are grouped together. This structure is crucial for researchers who may wish to perform speaker-specific analysis or adaptation.

  The canonical directory structure is as follows:
  ```
  data_root/
  ├── split_name/  // e.g., dev-clean, test-clean, train-clean-100
  │   ├── speaker_id/
  │   │   ├── chapter_id/
  │   │   │   ├── speaker_id-chapter_id-utterance_id.flac
  │   │   │   └── speaker_id-chapter_id.trans.txt
  ```
  Each chapter directory contains multiple `.flac` audio files, representing individual utterances, and a single `.trans.txt` file. This transcript file contains the corresponding text for all audio utterances within that chapter, mapping each audio filename to its ground-truth transcription.
])


=== #en([Standard Data Splits])

#en_std([
  The corpus is officially partitioned into several pre-defined splits, each serving a distinct purpose in the model development lifecycle. The primary splits are categorized as "clean" (less background noise and reverberation) and "other" (more challenging audio).

  #figure(
    table(
      columns: (2fr, 0.75fr, 1.25fr, 4fr),
      rows: 3em,
      align: (left+horizon, center+horizon, center+horizon, left+horizon),
      [*Split Name*], [*Hours*], [*Speakers*], [*Purpose*],
      [`train-clean-100`], [100], [251], [Initial training on clean speech.],
      [`train-clean-360`], [360], [921], [Extended training on clean speech.],
      [`train-other-500`], [500], [1166], [Training on more challenging audio.],
      [`dev-clean`], [5.4], [40], [Validation/development on clean speech.],
      [`dev-other`], [5.1], [39], [Validation/development on challenging audio.],
      [`test-clean`], [5.4], [40], [Final evaluation on unseen clean speech.],
      [`test-other`], [5.1], [40], [Final evaluation on unseen challenging audio.],
    ),
    caption: flex_captions(
      [Data splits available in the LibriSpeech corpus.],
      [LibriSpeech corpus data splits]
    )
  )

  This standardized partitioning allows for reproducible experiments and fair comparison between different ASR models and methodologies.
])

== #en([Data Augmentation for Speech: SpecAugment])

#en_std([
  A significant challenge in developing speech recognition models is ensuring they are robust enough to handle the variability of real-world audio. Differences in speaker accents, speaking rates, and background noise can significantly degrade performance. To address this, we incorporate *SpecAugment*, a simple but powerful data augmentation technique that operates directly on the model's input features.

  SpecAugment introduces distortions to the log-mel spectrogram of the audio, teaching the model to become invariant to these variations without altering the underlying transcript. This method is computationally cheap and can be applied on-the-fly during training. It consists of three primary augmentation policies:
])

=== #en([Time Warping])

#en_std([
  Time warping is a technique that distorts the spectrogram along the time axis. A random point along the time axis is selected, and all time steps to the left or right of that point are warped by a random distance. This simulates variations in speaking speed, forcing the model to learn features that are not dependent on a fixed temporal pattern.
])

=== #en([Frequency Masking])

#en_std([
  In frequency masking, a block of consecutive frequency channels is selected and masked with a value of zero. This is analogous to covering a horizontal band on the spectrogram. By doing this, we prevent the model from becoming overly reliant on specific frequency information, which might not always be present due to noise or speaker-specific vocal characteristics. This encourages the model to learn more distributed and robust features.
])

=== #en([Time Masking])

#en_std([
  Similar to frequency masking, time masking involves masking out a block of consecutive time steps. This is equivalent to covering a vertical band on the spectrogram. This technique makes the model more resilient to occlusions in the time domain, such as brief pauses, stutters, or short bursts of noise that might obscure a part of the speech signal.

  #linebreak()

  By applying these three transformations, SpecAugment creates a more diverse set of training examples from the existing data. This helps the model generalize better to unseen data, significantly improving its robustness and reducing the Word Error Rate (WER), especially in noisy or less-than-ideal conditions.

  #pagebreak()
])

== #en([The Listen, Attend, and Spell (LAS) Architecture])

#en_std([
  The Listen, Attend, and Spell (LAS) model is a neural network architecture designed to transcribe speech utterances directly into character sequences. It's an end-to-end system that learns all components of a speech recognizer jointly, moving away from traditional, disjointly trained DNN-HMM models.

  The core of LAS is a *sequence-to-sequence* framework composed of two primary modules: an encoder called the *listener* and a decoder called the *speller*. The listener processes the input audio, and the speller generates the corresponding text, character by character.

  The overall model seeks to map an input sequence of filter bank spectra, \ $x = (x_1, ..., x_T)$, to an output sequence of characters, $y = (y_1, ..., y_S)$. This is achieved by modeling the conditional probability using the chain rule:

  #math.equation(
    [
      $P(y|x) = product_i P(y_i|x, y_(<i))$
    ],
    block: true
  )

  The two main functions, $"Listen"$ and $"AttendAndSpell"$, transform the input signal $x$ into a high-level representation $h$ and then produce the final probability distribution over the character sequence.

  #math.equation(
    [
      $h = "Listen"(x)$\
      $P(y|x) = "AttendAndSpell"(h, y)$
    ],
    block: true
  )

  #pagebreak()

  The following figure visualizes the architecture of the LAS model. It shows the pyramidal structure of the listener and the connection of the output of the listener into the input of the speller.

  #figure(
    image("../media/LAS_Full-Architecture.png", width: 75%),
    kind: image,
    caption: flex_captions(
      [The end-to-end architecture of the Listen, Attend, and Spell (LAS) model, showing the flow from the listener to the speller.],
      [Full LAS Architecture]
    )
  )

  #pagebreak()
])

=== #en([Listener (Encoder)])

#en_std([
  The *listener* acts as the acoustic model encoder. Its primary role is to convert the low-level acoustic features from the input speech signal $x$ into a higher-level feature representation $h = (h_1, ..., h_U)$.

  A key challenge in speech recognition is the length of the input signal, which can be thousands of time steps long. A standard RNN would struggle to extract relevant information from such a long sequence. To address this, the LAS listener uses a specialized architecture called a *pyramidal Bidirectional Long Short-Term Memory* (pBLSTM) network.

  The pBLSTM's main purpose is to *reduce the temporal resolution* of the feature sequence, making the sequence $h$ shorter than the input $x$ (i.e., $U < T$). In each successive layer of the pBLSTM, the time resolution is reduced by a factor of $2$. This is accomplished by concatenating the outputs from consecutive time steps in one layer before feeding them as input to the next layer. The update rule for a pBLSTM layer $j$ at time step $i$ is defined as:

  #math.equation(
    [
      $h_i^j = "pBLSTM"(h_(i-1)^j, [h_(2i)^(j-1), h_(2i+1)^(j-1)])$
    ],
    block: true
  )

  In the paper's implementation, the listener stacks three pBLSTM layers on top of an initial BLSTM layer, reducing the final time resolution by a factor of $2^3 = 8$. This significant reduction in length not only helps the attention mechanism to focus more easily but also reduces the overall computational complexity of the model.

  #figure(
    image("../media/LAS_Listener-Architecture.png", width: 50%),
    kind: image,
    caption: flex_captions(
      [The Listener encoder architecture. It uses a pyramidal BLSTM to progressively reduce the temporal resolution of the input features $x$, producing a shorter, high-level feature sequence $h$ for the Speller.],
      [Listener Architecture]
    )
  )
])


=== #en([Speller (Decoder)])

#en_std([
  The *speller* is an *attention-based* decoder that receives the high-level feature representation $h$ from the listener and generates the final transcript one character at a time.

  At each output step $i$, the speller computes a probability distribution over all possible characters, conditioned on the listener's output $h$ and all previously generated characters $y_(<i)$. This process is managed by an Recurrent Neural Network (RNN) (specifically, a 2-layer LSTM) that maintains a decoder state $s_i$ and uses a context vector $c_i$ provided by the attention mechanism.

  The relationship between the decoder state $s_i$, context vector $c_i$, and the final character probability is as follows:

  #math.equation([
      $s_i = "RNN"(s_(i-1), y_(i-1), c_(i-1))$\
      $P(y_i|x, y_(<i)) = "CharacterDistribution"(s_i, c_i)$
    ],
    block: true
  )

  The attention mechanism generates a context vector $c_i$ at each step. This vector provides the speller with focused information from the relevant parts of the input audio needed to generate the next character.

  The mechanism works as follows:
  + For the current decoder state $s_i$, a scalar "energy" value $e_(i,u)$ is calculated for each time step $u$ of the listener's output $h$. This energy measures the alignment between the decoder's current state and each part of the encoded audio.
  + The energy values are normalized using a *softmax* function to create a probability distribution $alpha_i$, known as the *attention vector*.
  + The final context vector $c_i$ is computed as a weighted sum of the listener's feature vectors $h_u$, where the weights are the attention probabilities $alpha_(i,u)$.

  The equations governing this process are:

  #math.equation([
      $e_(i,u) = <phi(s_i), psi(h_u)>$\
      $alpha_(i,u) = (exp(e_(i,u))) / (sum_u exp(e_(i,u)))$\
      $c_i = sum_u alpha_(i,u) h_u$
    ],
    block: true
  )

  The equations above describe *content-based attention* which was improved upon with more modern techniques described in later sections.

  Here, $phi$ and $psi$ are small Multi-Layer Perceptrons (MLPs). This mechanism allows the speller to dynamically decide which frames of the audio to "attend" to when producing each character, creating a direct and learnable alignment between the audio and the transcript.

  #figure(
    image("../media/LAS_Speller-Architecture.png", width: 60%),
    kind: image,
    caption: flex_captions(
      [The attention-based Speller decoder. At each step, it uses the previous character $y$, its internal state $s$, and a context vector $c$ (generated by the attention mechanism over $h$) to produce the next character in the sequence.],
      [Speller Architecture]
    )
  )
])


== #en([Enhancing the Speller with Advanced Attention Mechanisms])

#en_std([
  The standard LAS model uses a content-based attention mechanism that allows the speller to focus on relevant parts of the encoded audio signal. While effective, this foundational mechanism can be significantly improved with more advanced techniques, each designed to solve a specific challenge in sequence-to-sequence tasks. This section explores two such enhancements that are crucial for building a state-of-the-art model.
])

=== #en([Improving Representational Power: Multi-Head Attention])

#en_std([
  First introduced in the paper "Attention Is All You Need," *Multi-Head Attention* enhances the model's ability to capture a wider range of features from the input sequence. Instead of performing a single attention calculation, this mechanism runs multiple attention processes—or "heads"—in parallel.

  Each head learns to focus on a different aspect or subspace of the input representation. For instance, in speech, one head might learn to focus on low-level phonetic features, while another might capture prosodic information like intonation or rhythm. The outputs of these parallel heads are then concatenated and linearly transformed to produce the final attention output. This structure allows the model to jointly attend to information from different representation subspaces at different positions, yielding a richer and more nuanced understanding of the input.
])

=== #en([Improving Alignment Consistency: Location-Aware Attention])

#en_std([
  For tasks like speech recognition where the input and output sequences align monotonically (i.e., in a consistent forward direction), a purely content-based attention mechanism can sometimes struggle. It may get stuck attending to the same portion of the audio, causing repeated words, or it might jump ahead, skipping parts of the input entirely.

  *Location-Aware Attention*, proposed in "Attention-Based Models for Speech Recognition," directly addresses this issue. It enhances the standard mechanism by making it aware of its past alignment choices. This is achieved by feeding the attention weights from the previous time step as an additional input when calculating the attention for the current time step. This simple but powerful modification forces the model to be aware of its "location" in the input sequence, strongly encouraging it to move forward consistently and preventing alignment errors.

  These two mechanisms are not mutually exclusive and can be combined to create a highly effective attention system. The final model developed for this project integrates both *Multi-Head* and *Location-Aware Attention*. This hybrid approach leverages the enhanced feature extraction capabilities of multiple heads while ensuring stable, consistent alignment through location-awareness, resulting in a more robust and accurate transcription model overall.
])

== #en([Model Training and Evaluation Principles])

#en_std([
  Training a neural network involves iteratively optimizing its parameters to minimize a defined error, while simultaneously measuring its performance on unseen data. This process relies on three key components: a loss function to quantify the error, an optimizer to guide the learning process, and evaluation metrics to measure performance.
])

=== #en([Loss Function: Sparse Categorical Cross-Entropy])

#en_std([
  A *loss function* calculates a value that represents the difference between the model's prediction and the ground-truth label. The goal of training is to minimize this value.

  For multi-class classification tasks like character prediction, the standard loss function is *Categorical Cross-Entropy*. It is effective when the model outputs a probability distribution (via a Softmax function) across all possible classes.

  This project uses a common variant called *Sparse Categorical Cross-Entropy*. This version is functionally identical but is designed for scenarios where the true labels are provided as integers (e.g., `[23, 4, 15]`) rather than one-hot encoded vectors (e.g., `[[0..1..0], [0..1..0]]`). This is more memory-efficient and perfectly suited for sequence-to-sequence tasks where the targets are integer character IDs.
])

=== #en([Optimizer: Adam])

#en_std([
  An *optimizer* is an algorithm that modifies the model's internal parameters (weights) in response to the output of the loss function. Its goal is to find a set of weights that results in the minimum possible loss.

  The *Adam (Adaptive Moment Estimation)* optimizer is a highly effective and widely used algorithm for training deep neural networks. It combines the advantages of two other popular optimizers, AdaGrad and RMSProp, by maintaining two separate moving averages of the gradients:
  + The first moment (the mean), which acts like momentum.
  + The second moment (the uncentered variance), which provides an adaptive, per-parameter learning rate.

  This combination allows Adam to converge quickly and perform robustly across a wide range of problems, making it a standard choice for complex models.
])


=== #en([Evaluation Metric: Character Error Rate (CER)])

#en_std([
  Character Error Rate (CER) measures errors at the character level. It is calculated by comparing the predicted character sequence to the ground-truth transcript and summing the number of substitutions (S), insertions (I), and deletions (D) required to transform one into the other. This sum is then normalized by the total number of characters in the reference transcript (N).

  #math.equation(
    [
      $"CER" = (S + D + I) / N$
    ],
    block: true
  )
])

=== #en([Evaluation Metric: Word Error Rate (WER)])

#en_std([
  Word Error Rate (WER) is the industry-standard evaluation metric for ASR systems. It operates identically to CER but at the word level, calculating the minimum number of word substitutions, deletions, and insertions required to match the reference sentence, normalized by the total number of words in the reference.

  #math.equation(
    [
      $"WER" = (S + D + I) / N$
    ],
    block: true
  )

  A lower WER indicates a more accurate transcription system.
])

== #en([Advanced Training and Inference Strategies])

#en_std([
  Beyond the core model architecture and basic training loop, several advanced strategies can be employed to enhance performance. This section covers techniques that address specific challenges in sequence generation, from improving the model's robustness during training to refining the quality of the final output during inference.
])

=== #en([Training Strategy: Scheduled Sampling])

#en_std([
  A significant discrepancy often exists between how a recurrent neural network is trained and how it is used for inference. Scheduled Sampling is a curriculum learning strategy designed to bridge this gap, making the model more robust to its own mistakes by gradually altering the training process.
])

==== #en([The Training-Inference Discrepancy])

#en_std([
  The core of the issue lies in the source of input to the decoder at each time step.

  - *Teacher Forcing*: The standard training method provides the ground-truth previous token, $y_(t-1)$, as input when predicting the next token, $y_t$. This is efficient and stable, but it creates a mismatch with inference, where the ground-truth is unavailable. The model is never trained to recover from its own errors.
  - *Free-Running*: The opposite approach is to feed the model's own previous prediction, $hat(y)_(t-1)$, as input during training. While this mirrors inference conditions perfectly, it is extremely difficult to train from scratch, as the initially random predictions can lead to chaotic inputs and prevent convergence. This method has been shown to yield poor performance.
])

==== #en([Bridging the Gap])

#en_std([
  Scheduled Sampling provides a middle ground by mixing these two approaches. At each time step, it randomly decides whether to feed the model the ground-truth token or a token sampled from its own output distribution.

  The probability that governs this choice is denoted as $epsilon$. It's important to note how this project's implementation defines this term. While the original paper by Bengio et al. defines $epsilon$ as the probability of selecting the *ground-truth* token, this project inversely defines $epsilon$ as the probability of selecting the *model's own predicted token*. The ground-truth is therefore chosen with a probability of $1 - epsilon$.

  The "scheduled" aspect comes from using a curriculum where the value of $epsilon$ is gradually increased during training. At the start, $epsilon$ is low, and the model relies on Teacher Forcing. As training progresses, $epsilon$ increases, forcing the model to rely more on its own predictions, thereby teaching it to be more robust and capable of correcting its own errors. The original paper proposes several schedules for this transition, such as Linear, Exponential, and Sigmoid increases.
])

==== #en([Impact on LAS])

#en_std([
  The authors of the "Listen, Attend, and Spell" paper demonstrated the practical benefits of this technique, which they call a "sampling trick". By training their model with a $10%$ probability of using a character sampled from its own output, they achieved a significant reduction in Word Error Rate (WER).

  The table below, using data from the original paper, compares the performance of the LAS model with and without sampling.

  #figure(
    table(
      columns: (3fr, 2fr, 2fr),
      align: (center, center, center),
      [*Model*], [*Clean WER*], [*Noisy WER*],
      [LAS (Baseline)], [$16.2%$], [$19.0%$],
      [LAS + Sampling], [$14.1%$], [$16.5%$],
    ),
    kind: table,
    caption: flex_captions(
      [WER comparison of the LAS model with and without Scheduled Sampling, as reported in the original paper.],
      [LAS vs. LAS + Sampling WER]
    )
  )

  This result shows that Scheduled Sampling is a valuable technique that leads to more robust and accurate sequence generation models in practice.
])


=== #en([Inference Strategy: Beam Search Decoding])

#en_std([
  During inference, the goal is to find the most accurate sequence from the vast number of possibilities the model can generate. A simple *greedy search*, which picks the single most likely character at each step, can lead to suboptimal results because an early mistake cannot be undone.

  A more effective method is *beam search decoding*. Instead of tracking only one path, beam search maintains a set number of the most probable partial sequences—the "beam". At each step, every sequence in the beam is expanded with all possible next characters, and only the highest-probability resulting sequences are kept for the next step. This allows the decoder to explore a much larger search space and recover from locally poor decisions, often resulting in a more accurate final transcript.
])


=== #en([Inference Strategy: Language Model Integration])

#en_std([
  Another approach to improve accuracy is to integrate external knowledge sources during decoding. Two common methods are dictionary constraints and language model rescoring.

  - *Dictionary Constraint*: One might use a dictionary to limit the search space to only valid words, correcting spelling errors. However, the LAS paper authors found this to have no impact on WER, as their model learned to spell correctly on its own.
  - *Language Model Rescoring*: A more effective technique is to use an external, independently trained language model to rescore the list of candidate sequences produced by beam search. The language model assesses the linguistic plausibility of each hypothesis and adjusts its score accordingly. This method has been shown to significantly improve final accuracy.
])

#pagebreak()
