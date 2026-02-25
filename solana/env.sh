#!/bin/bash
# Source this to get Solana/Anchor/Rust toolchain in PATH
export HOME="${HOME:-/Users/queenmab}"
export PATH="$HOME/.cargo/bin:$HOME/.local/share/solana/install/active_release/bin:$HOME/.avm/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:$PATH"

# NVM for node/yarn if needed
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
