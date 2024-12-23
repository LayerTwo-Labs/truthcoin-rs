# Truthcoin-rs

A Rust implementation of the Truthcoin protocol, providing a decentralized oracle and prediction market system.

## Features

- Outcomes resolve without a central server (ie, "oracle problem" is solved)
- Trade via MSR (no counterparty, just one signed message)
- Anyone can make markets on anything, or trade on anything

## Getting Started

### Prerequisites

- Rust 1.70 or higher
- Cargo package manager

### Installation

```bash
git clone git@github.com:LayerTwo-Labs/truthcoin-rs.git
cd truthcoin-rs
cargo build --release
```

### Running Tests

```bash
cargo test
```

## Project Structure

- `src/chain_objects/` - Blockchain-related data structures
- `src/crypto/` - Cryptographic operations
- `src/custom_math/` - Mathematical utilities and consensus algorithms
- `src/r-code/` - R language reference implementations

## Documentation

Generate and view the documentation locally:

```bash
cargo doc --no-deps --open
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original Truthcoin whitepaper
- Rust community and crate authors
