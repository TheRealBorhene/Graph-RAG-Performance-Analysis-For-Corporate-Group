import os
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

GRAPH_PATH = "../data/graph_output.json"

# Neo4j does not allow spaces in labels — map entity types to valid label names
LABEL_MAP = {
    "Parent":           "Parent",
    "Subsidiary":       "Subsidiary",
    "Geography":        "Geography",
    "Business Segment": "BusinessSegment",
    "Financial Item":   "FinancialItem",
    "BalanceSheet":     "BalanceSheet",
    "IncomeStatement":  "IncomeStatement",
    "CashFlow":         "CashFlow",
}


# ─────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────
def connect() -> GraphDatabase.driver:
    uri      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    user     = os.getenv("NEO4J_USER",     "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    print(f"Connected to Neo4j at {uri}")
    return driver


# ─────────────────────────────────────────────
# CLEAR
# ─────────────────────────────────────────────
def clear_graph(session) -> None:
    session.run("MATCH (n) DETACH DELETE n")
    print("Existing graph cleared.")


# ─────────────────────────────────────────────
# NODES
# ─────────────────────────────────────────────
def create_nodes(session, entities: list[dict]) -> None:
    skipped = 0
    for entity in entities:
        label = LABEL_MAP.get(entity["type"])
        if not label:
            print(f"  [WARN] Unknown entity type '{entity['type']}' — skipping '{entity['name']}'")
            skipped += 1
            continue

        # Merge the node, then set standard metadata fields.
        # For statement nodes, also spread their properties dict onto the node
        # using SET n += $props (adds/updates without removing existing keys —
        # this is what allows cross-chunk partial statement merging to accumulate).
        props = entity.get("properties") or {}
        session.run(
            f"MERGE (n:Entity:{label} {{name: $name}}) "
            f"SET n += $props "
            f"SET n.type        = $type, "
            f"    n.chunk_id    = $chunk_id, "
            f"    n.file        = $file, "
            f"    n.page_number = $page_number",
            name        = entity["name"],
            props       = props,
            type        = entity["type"],
            chunk_id    = entity.get("chunk_id"),
            file        = entity.get("file"),
            page_number = entity.get("page_number")
        )

    stored = len(entities) - skipped
    print(f"  Nodes   : {stored} stored, {skipped} skipped (unknown type)")


# ─────────────────────────────────────────────
# RELATIONSHIPS
# ─────────────────────────────────────────────
def create_relationships(session, relationships: list[dict]) -> None:
    created = 0
    skipped = 0

    for rel in relationships:
        rel_type = rel.get("type")
        if not rel_type:
            skipped += 1
            continue

        result = session.run(
            f"MATCH (a:Entity {{name: $source}}), (b:Entity {{name: $target}}) "
            f"MERGE (a)-[r:{rel_type}]->(b) "
            f"SET r.property    = COALESCE($property, r.property), "
            f"    r.chunk_id    = $chunk_id, "
            f"    r.page_number = $page_number",
            source      = rel["source"],
            target      = rel["target"],
            property    = rel.get("property"),
            chunk_id    = rel.get("chunk_id"),
            page_number = rel.get("page_number")
        )
        summary = result.consume()

        # count as created if at least one relationship or property was written
        if summary.counters.relationships_created > 0 or summary.counters.properties_set > 0:
            created += 1
        else:
            skipped += 1

    print(f"  Relations: {created} stored, {skipped} skipped (nodes not found or already existed)")


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────
def store_graph(graph: dict) -> None:
    entities      = graph.get("entities", [])
    relationships = graph.get("relationships", [])

    print(f"\nGraph to store:")
    print(f"  Entities      : {len(entities)}")
    print(f"  Relationships : {len(relationships)}\n")

    driver = connect()

    with driver.session() as session:
        clear_graph(session)
        print("\nCreating nodes...")
        create_nodes(session, entities)
        print("\nCreating relationships...")
        create_relationships(session, relationships)

    driver.close()

    print(f"\n{'─' * 50}")
    print(f"Graph stored successfully in Neo4j.")
    print(f"{'─' * 50}")


# ─────────────────────────────────────────────
# RUN DIRECTLY: load graph_output.json → Neo4j
# usage: python graph.py
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Loading graph from {GRAPH_PATH}...")
    with open(GRAPH_PATH, "r", encoding="utf-8") as f:
        graph = json.load(f)

    store_graph(graph)
