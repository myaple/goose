#!/usr/bin/env bash
set -eu

##############################################################################
# Goose CLI Install Script
#
# This script downloads the latest stable 'goose' CLI binary from GitHub releases
# and installs it to your system.
#
# Supported OS: macOS (darwin), Linux
# Supported Architectures: x86_64, arm64
#
# Usage:
#   curl -fsSL https://github.com/block/goose/releases/download/stable/download_cli.sh | bash
#
# Environment variables:
#   GOOSE_BIN_DIR  - Directory to which Goose will be installed (default: $HOME/.local/bin)
#   GOOSE_PROVIDER - Optional: provider for goose
#   GOOSE_MODEL    - Optional: model for goose
#   CANARY         - Optional: if set to "true", downloads from canary release instead of stable
#   CONFIGURE      - Optional: if set to "false", disables running goose configure interactively
#   ** other provider specific environment variables (eg. DATABRICKS_HOST)
##############################################################################

# --- 1) Check for dependencies ---
# Check for curl
if ! command -v curl >/dev/null 2>&1; then
  echo "Error: 'curl' is required to download Goose. Please install curl and try again."
  exit 1
fi

# Check for tar
if ! command -v tar >/dev/null 2>&1; then
  echo "Error: 'tar' is required to download Goose. Please install tar and try again."
  exit 1
fi


# --- 2) Variables ---
REPO="block/goose"
OUT_FILE="goose"
GOOSE_BIN_DIR="${GOOSE_BIN_DIR:-"$HOME/.local/bin"}"
RELEASE="${CANARY:-false}"
RELEASE_TAG="$([[ "$RELEASE" == "true" ]] && echo "canary" || echo "stable")"
CONFIGURE="${CONFIGURE:-true}"

# --- 3) Detect OS/Architecture ---
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

case "$OS" in
  linux|darwin) ;;
  *)
    echo "Error: Unsupported OS '$OS'. Goose currently only supports Linux and macOS."
    exit 1
    ;;
esac

case "$ARCH" in
  x86_64)
    ARCH="x86_64"
    ;;
  arm64|aarch64)
    # Some systems use 'arm64' and some 'aarch64' – standardize to 'aarch64'
    ARCH="aarch64"
    ;;
  *)
    echo "Error: Unsupported architecture '$ARCH'."
    exit 1
    ;;
esac

# Build the filename and URL for the stable release
if [ "$OS" = "darwin" ]; then
  FILE="goose-$ARCH-apple-darwin.tar.bz2"
else
  FILE="goose-$ARCH-unknown-linux-gnu.tar.bz2"
fi

DOWNLOAD_URL="https://github.com/$REPO/releases/download/$RELEASE_TAG/$FILE"

# --- 4) Download & extract 'goose' binary ---
echo "Downloading $RELEASE_TAG release: $FILE..."
if ! curl -sLf "$DOWNLOAD_URL" --output "$FILE"; then
  echo "Error: Failed to download $DOWNLOAD_URL"
  exit 1
fi

# Create a temporary directory for extraction
TMP_DIR="/tmp/goose_install_$RANDOM"
if ! mkdir -p "$TMP_DIR"; then
  echo "Error: Could not create temporary extraction directory"
  exit 1
fi
# Clean up temporary directory
trap 'rm -rf "$TMP_DIR"' EXIT

echo "Extracting $FILE to temporary directory..."
set +e  # Disable immediate exit on error
tar -xjf "$FILE" -C "$TMP_DIR" 2> tar_error.log
tar_exit_code=$?
set -e  # Re-enable immediate exit on error

# Check for tar errors
if [ $tar_exit_code -ne 0 ]; then
  if grep -iEq "missing.*bzip2|bzip2.*missing|bzip2.*No such file|No such file.*bzip2" tar_error.log; then
    echo "Error: Failed to extract $FILE. 'bzip2' is required but not installed. See details below:"
  else
    echo "Error: Failed to extract $FILE. See details below:"
  fi
  cat tar_error.log
  rm tar_error.log
  exit 1
fi
rm tar_error.log

rm "$FILE" # clean up the downloaded tarball

# Make binary executable
chmod +x "$TMP_DIR/goose"

# --- 5) Install to $GOOSE_BIN_DIR ---
if [ ! -d "$GOOSE_BIN_DIR" ]; then
  echo "Creating directory: $GOOSE_BIN_DIR"
  mkdir -p "$GOOSE_BIN_DIR"
fi

echo "Moving goose to $GOOSE_BIN_DIR/$OUT_FILE"
mv "$TMP_DIR/goose" "$GOOSE_BIN_DIR/$OUT_FILE"

# Also move temporal-service if it exists (for scheduling functionality)
if [ -f "$TMP_DIR/temporal-service" ]; then
  echo "Moving temporal-service to $GOOSE_BIN_DIR/temporal-service"
  mv "$TMP_DIR/temporal-service" "$GOOSE_BIN_DIR/temporal-service"
  chmod +x "$GOOSE_BIN_DIR/temporal-service"
fi

# skip configuration for non-interactive installs e.g. automation, docker
if [ "$CONFIGURE" = true ]; then
  # --- 6) Configure Goose (Optional) ---
  echo ""
  echo "Configuring Goose"
  echo ""
  "$GOOSE_BIN_DIR/$OUT_FILE" configure
else
  echo "Skipping 'goose configure', you may need to run this manually later"
fi

# --- 7) Check PATH and give instructions if needed ---
if [[ ":$PATH:" != *":$GOOSE_BIN_DIR:"* ]]; then
  echo ""
  echo "Warning: Goose installed, but $GOOSE_BIN_DIR is not in your PATH."
  echo "Add it to your PATH by editing ~/.bashrc, ~/.zshrc, or similar:"
  echo "    export PATH=\"$GOOSE_BIN_DIR:\$PATH\""
  echo "Then reload your shell (e.g. 'source ~/.bashrc', 'source ~/.zshrc') to apply changes."
  echo ""
fi
