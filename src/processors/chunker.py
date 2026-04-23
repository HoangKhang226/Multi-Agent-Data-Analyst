from src.utils.logger import logger
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core import Document as LlamaDocument


class Chunker:
    """Splits document text into retrieval-ready hierarchical nodes using LlamaIndex.

    Automatically creates a parent-child relationship tree for context-aware retrieval.
    """

    def __init__(self):
        # Configure chunk sizes: [Parent, intermediate, Child]
        # Example: 2048 -> 512 -> 128
        self.node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 512, 128]
        )
        logger.debug("LlamaIndex HierarchicalNodeParser initialized")

    def chunk(self, docs: list[LlamaDocument]):
        """Split a list of LlamaDocuments into a hierarchical node tree.

        Args:
            docs: List of LlamaIndex Document objects.

        Returns:
            A tuple (nodes, leaf_nodes):
              - nodes: All nodes in the hierarchy (for storage and relationship context).
              - leaf_nodes: Only the smallest chunks (for semantic search).
        """
        if not isinstance(docs, list):
            docs = [docs]

        logger.info(f"Chunking {len(docs)} documents into a hierarchical structure...")
        
        # Automatically generate Parent - Child graph nodes
        nodes = self.node_parser.get_nodes_from_documents(docs)
        
        # We store leaf nodes for search, but keep all nodes to preserve the hierarchy
        leaf_nodes = get_leaf_nodes(nodes)
        
        logger.info(
            f"Hierarchical chunking completed — {len(nodes)} total nodes, "
            f"{len(leaf_nodes)} leaf nodes produced"
        )
        return nodes, leaf_nodes