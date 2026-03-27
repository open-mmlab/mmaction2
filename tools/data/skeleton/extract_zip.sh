#!/bin/bash

# Check if correct number of arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 <zip_file> <destination_directory>"
    exit 1
fi

# Assign arguments to variables
ZIP_FILE="$1"
DEST_DIR="$2"

# Check if zip file exists
if [ ! -f "$ZIP_FILE" ]; then
    echo "Error: Zip file '$ZIP_FILE' not found"
    exit 1
fi

# Check if destination directory exists, create it if it doesn't
if [ ! -d "$DEST_DIR" ]; then
    echo "Creating destination directory '$DEST_DIR'"
    mkdir -p "$DEST_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create destination directory"
        exit 1
    fi
fi

# Extract the zip file to destination
echo "Extracting '$ZIP_FILE' to '$DEST_DIR'"
unzip -q "$ZIP_FILE" -d "$DEST_DIR"

# Check if extraction was successful
if [ $? -eq 0 ]; then
    echo "Extraction completed successfully"
else
    echo "Error: Failed to extract zip file"
    exit 1
fi