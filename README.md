# MojoCNN: LeNet-5 Implementation from Scratch (WIP)

A high-performance implementation of a GPT-2-like LLM built entirely from scratch in Mojo🔥 with a custom tokenizer, CPU ops, logging, allocators, and the like. Forward pass is done for CPU at the time you're reading this. I still need to finish training, optimizers, logging, and clean things up.

## Project Motivation

This project was undertaken as a deep learning exercise to:
- **Learn LLMs from first principles** by implementing every component from scratch
- **Explore Mojo**, a cutting-edge systems programming language designed for AI workloads
- **Build custom kernels** without relying on existing ML libraries or frameworks
- **Achieve competitive performance** through low-level optimization and manual memory management

## Architecture

This implementation features a GPT-2-like architecture:
- BPE tokenization
- Embedding Layer
- Single-Headed Transformer Blocks with Naive Attention
- LayerNorm
- Output logits
- GeLU activations

## Project Structure

Inside the /src/ ...

```
├── activation_fn.mojo              # Activation functions
├── arena.mojo              # BumpArenaAllocator
├── attention.mojo              # LLM Implementation
├── cliparser.mojo              # Simple CLI parsing of argv
├── helpers.mojo              # Pretty printing, progress bars, buffer comparisons, etc.
├── logger.mojo              # Data logging for training and inference
├── main.mojo              # Tokenize data set, feed into model in training loop, log, display results
├── ops.mojo              # CPU kernels for forward and backward pass
├── token_chunks.mojo              # Data structure for fast tokenization
├── tokenizer.mojo              # BPE Tokenizer
├── graph*.sh              # Create a flamegraph.svg
├── cleangraph.sh              # graph*.sh cleanup
├── datasets/              # PyTorch reference implementation
│   └── shakespeare.txt         # Kaparthy reference
├── logs/
│   └── *.csv              # Performance benchmarking results
├── models/
│   └── bpe*.tok           # BPE Tokenizer
│   └── vocab*.txt         # Tokenizer vocabs for displaying on webtool
```

## Technical Implementation

### Custom Components Built from Scratch
- **Memory Management**: Manual allocation using a custom bump arena allocator
- **Kernels**: Hand-written kernels in Mojo for all operations
- **Data Pipeline**: Custom data loader with BPE-trained custom tokenizer
- **Forward Pass**: Complete inference and training pipeline with logging

### Key Features
- Zero external ML library dependencies
- Custom memory management and kernel execution
- Custom logging for training and testing

## Getting Started

### Prerequisites
- Mojo 26.1+
- Pixi package manager

### Installation & Usage

```bash
# Install dependencies
pixi shell

# CPU training and inference
mojo main.mojo

# Build executable with optional --help
mojo build main.mojo
```

## Current Limitations & Future Work

### Known Limitations
- GPU not implemented
- Backward pass is WIP
- Optimizer is WIP

### Planned Improvements
- Stay tuned

## Contributing

This is primarily an educational project, but suggestions and discussions about optimization techniques or Mojo best practices are welcome!

## Acknowledgments

- Built with [Mojo🔥](https://www.modular.com/mojo) by Modular
