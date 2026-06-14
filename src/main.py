import json
from pathlib import Path
from dotenv import load_dotenv
from loader import load_document
from chunker import chunk_document, embed_and_store, fetch_chunks_from_qdrant
from extractor import extract_graph, merge_graph, enrich_financial_items, drop_orphaned_financial_items
from graph import store_graph

load_dotenv()

DEBUG     = True
file_path = "../data/LoewsCompany.md"

# step 1: load the document
text = load_document(file_path)

# step 2: chunk the document
chunks = chunk_document(text, source=Path(file_path).name)
print(f"Total chunks: {len(chunks)}")

if DEBUG:
    output_path = "../data/chunks_output.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(f"── Chunk {chunk['metadata']['chunk_id']} ──\n")
            f.write(f"Section  : {chunk['metadata']['section']}\n")
            f.write(f"File     : {chunk['metadata']['file']}\n")
            f.write(f"Text     :\n{chunk['text']}\n")
            f.write("\n" + "─" * 60 + "\n\n")
    print(f"Chunks saved to {output_path}")

# step 3: embed and store in qdrant
collection_name = Path(file_path).stem
client = embed_and_store(chunks, collection_name=collection_name)

# step 4: fetch chunks directly from qdrant
chunks_from_qdrant = fetch_chunks_from_qdrant(client, collection_name)

# step 5: filter out boilerplate chunks
relevant_chunks = [c for c in chunks_from_qdrant if c["metadata"]["section"] != "no section"]
filtered_count  = len(chunks_from_qdrant) - len(relevant_chunks)

print(f"Filtered out {filtered_count} boilerplate chunks")
print(f"Relevant chunks for extraction: {len(relevant_chunks)} out of {len(chunks_from_qdrant)}")

# step 6: extract entities and relationships
graph = extract_graph(relevant_chunks)

# step 7: deduplicate entities and relationships across chunks
graph = merge_graph(graph)

# step 7.5: enrich Financial Item nodes — rename from raw values to 'Label: value'
#           and simplify relationship properties to just the fiscal year
graph["entities"], graph["relationships"] = enrich_financial_items(
    graph["entities"], graph["relationships"]
)

# step 7.75: drop Financial Items with no relationship pointing to them
graph["entities"] = drop_orphaned_financial_items(graph["entities"], graph["relationships"])

# step 8: save the graph result
graph_output_path = "../data/graph_output.json"
with open(graph_output_path, "w", encoding="utf-8") as f:
    json.dump(graph, f, indent=2, ensure_ascii=False)

print(f"Graph saved to {graph_output_path}")

# step 9: store graph in Neo4j
store_graph(graph)