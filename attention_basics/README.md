# Attention Mechanisms

[Detailed Explaination Video](https://www.youtube.com/watch?v=NCn97L5WbCY)

Attention mechanisms focuses on different parts of the image to generate translation of the images to describe an image. They are generally adopted in sequence to sequence model and used for applications like Text Sentiment Analysis.

## Sequence to Sequence Models
Before we jump into learning about attention models, let's recap what you've learned about sequence to sequence models. We know that RNNs excel at using and generating sequential data, and sequence to sequence models can be used in a variety of applications!

[Seq to Seq part 1](https://www.youtube.com/watch?v=tDJBDwriJYQ)

[Seq to Seq part 2](https://www.youtube.com/watch?v=dkHdEAJnV_w)

[Seq to Seq part 3](https://www.youtube.com/watch?v=MRPHIPR0pGE)

A Seq to Seq model usually consist of an encoder and decoder. Encoder turns the input into a context vector which is then sent to decoder to output an output sequence. Typically, the encoder and decoders are both RCNNs, usually LSTM cells. 

## Encoders and Decoders
The encoder and decoder do not have to be RNNs; they can be CNNs too!

In the example above, an LSTM is used to generate a sequence of words; LSTMs "remember" by keeping track of the input words that they see and their own hidden state.

In computer vision, we can use this kind of encoder-decoder model to generate words or captions for an input image or even to generate an image from a sequence of input words. We'll focus on the first case: generating captions for images, and you'll learn more about caption generation in the next lesson. For now know that we can input an image into a CNN (encoder) and generate a descriptive caption for that image using an LSTM (decoder).

### Encoder
[Detailed Explaination Video](https://www.youtube.com/watch?v=IctAnMaVUKc)

Encoder outputs all the hidden vectors as context vector. Each vector focus on each word input with a little bit of the previous word encoperated. 

#### Attention Encoder
[Detailed Explaination Video](https://www.youtube.com/watch?v=sphe9LDT4rA)

### Decoder
[Detailed Explaination Video](https://www.youtube.com/watch?v=DJxiPd585GY)

At every time step, the attention decoder focus on each part of the context vector. It learns this during training phase. 

#### Attention Decoder
[Detailed Explaination Video](https://www.youtube.com/watch?v=5mMz6nN9_Ss)

## Attention Methods
[Detailed Explaination Video](https://www.youtube.com/watch?v=2eqIUDjefNg)

### Multiplicatiive Attention
[Detailed Explaination Video](https://www.youtube.com/watch?v=1-OwCgrx1eQ)

### Additive Attention
[Detailed Explaination Video](https://www.youtube.com/watch?v=93VfVWZ-IvY)

### Transformer and Self-Attention
[Detailed Explaination Video Part 1](https://www.youtube.com/watch?v=VmsR9FVpQiM)
[Detailed Explaination Video Part 2](https://www.youtube.com/watch?v=F-XN72bQiMQ)


