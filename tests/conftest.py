"""
Shared pytest fixtures and configuration.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from langchain_core.documents import Document


# ─── sample documents ─────────────────────────────────────────────────────────


SAMPLE_ALGO_TEXT = textwrap.dedent("""\
    BFS and DFS Graph Traversal

    Breadth-First Search (BFS) explores vertices level by level using a queue.
    Starting from a source vertex, it visits all neighbours before moving deeper.
    Time complexity: O(V + E) where V = vertices and E = edges.

    Depth-First Search (DFS) explores as far as possible before backtracking.
    It uses a stack (or recursion). Also O(V + E).

    Java Implementation (BFS):
    void bfs(int start, List<List<Integer>> adj) {
        boolean[] visited = new boolean[adj.size()];
        Queue<Integer> queue = new LinkedList<>();
        visited[start] = true;
        queue.add(start);
        while (!queue.isEmpty()) {
            int v = queue.poll();
            for (int u : adj.get(v)) {
                if (!visited[u]) {
                    visited[u] = true;
                    queue.add(u);
                }
            }
        }
    }

    Max-Flow Min-Cut Theorem

    The maximum flow in a network equals the minimum capacity of any cut
    separating the source from the sink.
    Ford-Fulkerson algorithm finds the max flow by repeatedly finding
    augmenting paths using DFS or BFS.
""")

SAMPLE_ALGO_TEXT_2 = textwrap.dedent("""\
    Quadtree Data Structure

    A quadtree is a tree data structure where each internal node has exactly
    four children, used to partition a 2D space recursively.

    Insertion into a quadtree:
    1. If the node is a leaf and has capacity, insert the point.
    2. If the node is at capacity, subdivide into four quadrants.
    3. Recurse into the appropriate quadrant.

    Time complexity of insertion: O(log n) on average for uniformly distributed points.
""")


@pytest.fixture
def sample_document() -> Document:
    """A single Document mimicking a PDF page."""
    return Document(
        page_content=SAMPLE_ALGO_TEXT,
        metadata={
            "source": "/data/raw/algorithms.pdf",
            "filename": "algorithms.pdf",
            "page": 1,
            "total_pages": 3,
            "file_type": "pdf",
        },
    )


@pytest.fixture
def sample_documents() -> list[Document]:
    """Two Documents from different sources."""
    return [
        Document(
            page_content=SAMPLE_ALGO_TEXT,
            metadata={
                "source": "/data/raw/algorithms.pdf",
                "filename": "algorithms.pdf",
                "page": 1,
                "total_pages": 3,
                "file_type": "pdf",
            },
        ),
        Document(
            page_content=SAMPLE_ALGO_TEXT_2,
            metadata={
                "source": "/data/raw/data_structures.pdf",
                "filename": "data_structures.pdf",
                "page": 5,
                "total_pages": 10,
                "file_type": "pdf",
            },
        ),
    ]


@pytest.fixture
def tmp_pdf(tmp_path: Path) -> Path:
    """
    Create a PDF with extractable text content for loader tests.

    Uses reportlab if available (produces a real text-layer PDF), otherwise
    falls back to creating a text file disguised as .pdf — which lets loaders.py
    tests run but will fail load_pdf (suitable for error-path tests only).
    Skips if neither is available.
    """
    try:
        from reportlab.pdfgen import canvas as rl_canvas

        pdf_path = tmp_path / "test_algo.pdf"
        c = rl_canvas.Canvas(str(pdf_path))
        c.drawString(72, 750, "BFS Algorithm")
        c.drawString(72, 730, "Time complexity: O(V + E)")
        c.drawString(72, 710, "Uses a queue for level-order traversal.")
        c.save()
        return pdf_path
    except ImportError:
        pass

    try:
        # Fallback: write a minimal valid PDF manually (text-layer)
        pdf_path = tmp_path / "test_algo.pdf"
        pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 44>>
stream
BT /F1 12 Tf 72 750 Td (BFS Algorithm) Tj ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000274 00000 n
0000000370 00000 n
trailer<</Size 6/Root 1 0 R>>
startxref
451
%%EOF"""
        pdf_path.write_bytes(pdf_content)
        return pdf_path
    except Exception:
        pytest.skip("Cannot create a PDF with extractable text in this environment")


@pytest.fixture
def tmp_text_file(tmp_path: Path) -> Path:
    """Create a plain text file with algorithm content."""
    path = tmp_path / "algorithms.txt"
    path.write_text(SAMPLE_ALGO_TEXT, encoding="utf-8")
    return path


@pytest.fixture
def tmp_md_file(tmp_path: Path) -> Path:
    """Create a markdown file with algorithm content."""
    path = tmp_path / "algorithms.md"
    path.write_text(SAMPLE_ALGO_TEXT_2, encoding="utf-8")
    return path
