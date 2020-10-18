# Long Short Term Memory (LSTM)

## RNN vs LSTM
[Detailed Explaination Video](https://www.youtube.com/watch?v=70MgF-IwAr8)

RNN have problem storing long term memory so LSTM tries to address this.

## Basics of LSTM
[Detailed Explaination Video](https://www.youtube.com/watch?v=gjb68a4XsqE)

## Architecture of LSTM
[Detailed Explaination Video](https://www.youtube.com/watch?v=ycwthhdx8ws)

The LSTM has a more complex cell where there's more gates in the cells

### The Learn Gate
[Detailed Explaination Video](https://www.youtube.com/watch?v=aVHVI7ovbHY)

Learn Gate takes the short term memory and the event and combines them, but it also ignore some of it and produce an output. This is all done using the tanh activation function and multiplied by an ignoring factor.

### The Forget Gate
[Detailed Explaination Video](https://www.youtube.com/watch?v=iWxpfxLUPSU)

Forget gate takes the long term memory and decides what to forget. It takes the Long term memory at a previous time and multiple it by a forget factor.

### The Remember Gate
[Detailed Explaination Video](https://www.youtube.com/watch?v=0qlm86HaXuU)

The remeber gate adds the output from forget gate and learn gate and adds them. 

### The Use Gate
[Detailed Explaination Video](https://www.youtube.com/watch?v=5Ifolm1jTdY)

Use the long term memory and short term memory and produce a new long term memory. Takes the output of forget gate and applies tanh and also applies sigmoid to short term memory then multiplies them.

### All together
[Detailed Explaination Video](https://www.youtube.com/watch?v=IF8FlKW-Zo0)

## Sequences of Data
[Detailed Explaination Video](https://www.youtube.com/watch?v=pdSr5F-9qE0)

The core reason that recurrent neural networks are exciting is that they allow us to operate over sequences of vectors: sequences in the input, the output, or in some cases, both!

Most RNN's expect to see a sequence of data in a fixed batch size, much like we've seen images processed in fixed batch sizes. The batches of data affect how the hidden state of an RNN train, so next you'll learn more about batching data.
