import os
import json


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "/app/app/rag/documents/ecommerce_knowledge.txt"

OUTPUT_DIR = "/app/app/rag/data"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "chunks.jsonl"
)

# Maximum number of words in a chunk
CHUNK_SIZE = 80

# Number of overlapping words between chunks
OVERLAP = 15


# ============================================================
# Load document
# ============================================================

def load_document(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


# ============================================================
# Split document into paragraphs
# ============================================================

def split_paragraphs(text):

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    return paragraphs


# ============================================================
# Create chunks
# ============================================================

def create_chunks(paragraphs):

    chunks = []

    chunk_id = 1

    for paragraph in paragraphs:

        words = paragraph.split()

        # ----------------------------------------------------
        # If paragraph is small, keep it as one chunk
        # ----------------------------------------------------

        if len(words) <= CHUNK_SIZE:

            chunks.append(
                {
                    "chunk_id": f"chunk-{chunk_id:03d}",
                    "source": os.path.basename(INPUT_FILE),
                    "text": paragraph,
                }
            )

            chunk_id += 1

            continue

        # ----------------------------------------------------
        # If paragraph is large, split it with overlap
        # ----------------------------------------------------

        start = 0

        while start < len(words):

            end = min(
                start + CHUNK_SIZE,
                len(words)
            )

            chunk_words = words[start:end]

            chunks.append(
                {
                    "chunk_id": f"chunk-{chunk_id:03d}",
                    "source": os.path.basename(INPUT_FILE),
                    "text": " ".join(chunk_words),
                }
            )

            chunk_id += 1

            # Stop when we reached the end
            if end == len(words):
                break

            # Move forward while keeping overlap
            start = end - OVERLAP

    return chunks


# ============================================================
# Save chunks as JSONL
# ============================================================

def save_chunks(chunks, path):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        for chunk in chunks:

            file.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False
                )
                + "\n"
            )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("RAG CHUNKING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load document
    # --------------------------------------------------------

    print(
        f"Reading document: {INPUT_FILE}"
    )

    document = load_document(
        INPUT_FILE
    )

    print(
        f"Document size: "
        f"{len(document.split())} words"
    )

    # --------------------------------------------------------
    # Split into paragraphs
    # --------------------------------------------------------

    paragraphs = split_paragraphs(
        document
    )

    print(
        f"Paragraphs found: "
        f"{len(paragraphs)}"
    )

    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    chunks = create_chunks(
        paragraphs
    )

    print(
        f"Chunks created: "
        f"{len(chunks)}"
    )

    # --------------------------------------------------------
    # Save chunks
    # --------------------------------------------------------

    save_chunks(
        chunks,
        OUTPUT_FILE
    )

    print(
        f"Saved to: "
        f"{OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Show sample chunks
    # --------------------------------------------------------

    print("\nSample chunks:")
    print("-" * 60)

    for chunk in chunks[:5]:

        print(
            f"\n{chunk['chunk_id']}"
        )

        print(
            f"Source: {chunk['source']}"
        )

        print(
            f"Text: {chunk['text'][:250]}"
        )

    print("\n" + "=" * 60)
    print("CHUNKING COMPLETED")
    print("=" * 60)