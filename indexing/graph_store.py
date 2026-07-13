from neo4j import Driver, GraphDatabase

from config import settings


def get_neo4j_driver() -> Driver:
    return GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )


def ensure_schema(driver: Driver) -> None:
    with driver.session() as session:
        session.run(
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
            "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT ref_key_unique IF NOT EXISTS "
            "FOR (r:Ref) REQUIRE r.ref_key IS UNIQUE"
        )
