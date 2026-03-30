import pandas as pd
import sys
import os
import time
import ast
import json
import logging
import numpy as np
from tqdm import tqdm
from neo4j import GraphDatabase
import chromadb
from google import genai
from observability import trace_tool

# Add Data_Dictionary to path for imports
sys.path.append(os.path.join(os.getcwd(), 'Data_Dictionary'))

try:
    from config import API_KEYS, MODELS
    import gemini_client
except ImportError:
    print("Warning: Could not import API_KEYS/gemini_client. Using placeholders.")
    API_KEYS = ["YOUR_API_KEY"]
    gemini_client = None

# --- Configuration (Adapted from Notebook) ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "data_dict1"
CHROMA_DB_PATH = "./chroma_db"
CSV_PATH = os.path.join(os.getcwd(), 'Data_Dictionary', 'data_dictionary_enriched.csv')
DATALAKE_PATH = os.path.join(os.getcwd(), 'Data_Dictionary', 'Datalake')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class RAGRetriever:
    """
    Retrieves relevant metadata context using Neo4j Knowledge Graph and GraphRAG.
    Implements logic from KG_Construction_Complete.ipynb.
    """
    
    def __init__(self):
        try:
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            # Test connection
            self.driver.verify_connectivity()
        except Exception as e:

            self.driver = None

        try:
            self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        except Exception as e:

            self.chroma_client = None
        
        # Initialize Gemini Client for embeddings (Rotation Logic)
        self.api_keys = API_KEYS
        self.current_key_index = 0
        if self.api_keys and self.api_keys[0] != "YOUR_API_KEY":
            try:
                self.sdk_client = genai.Client(api_key=self.api_keys[self.current_key_index])
            except:
                self.sdk_client = None
        else:
             self.sdk_client = None
        
        # Ensure Graph is built
        if self.driver:
            self._ensure_graph_initialized()

    def close(self):
        if self.driver:
            self.driver.close()

    def _get_embedding_with_rotation(self, text):
        """Fetches embeddings with automatic API key rotation on quota limits."""
        if not self.sdk_client:
            # Fallback to gemini_client if SDK not ready or failover needed
            if gemini_client:
                return gemini_client.get_embedding(text)
            return []

        while self.current_key_index < len(self.api_keys):
            try:
                result = self.sdk_client.models.embed_content(
                    model="gemini-embedding-001", 
                    contents=text
                )
                return result.embeddings[0].values
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "quota" in err_str or "limit" in err_str:
                    self.current_key_index += 1
                    if self.current_key_index < len(self.api_keys):

                        self.sdk_client = genai.Client(api_key=self.api_keys[self.current_key_index])
                        continue

                return [] 
        return []

    def _ensure_graph_initialized(self):
        """Checks if Neo4j graph exists, if not, runs construction pipeline."""
        try:
            with self.driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as count").single()
                count = result["count"]
                
            if count > 0:

                return


            if not os.path.exists(CSV_PATH):

                return

            df = pd.read_csv(CSV_PATH)


            # 1. Base Graph Construction
            with self.driver.session() as session:
                session.execute_write(self._create_base_graph_tx, df)


            # 2. Semantic Matching & Verification (Simplified for runtime)
            self._run_semantic_matching_and_verification(df)

            # 3. Create Vector Index
            self._create_vector_index(df)

        except Exception as e:
            pass

    def _create_base_graph_tx(self, tx, df):
        # Tables
        unique_tables = df[['Table Name', 'Table Description']].drop_duplicates('Table Name')
        for _, row in unique_tables.iterrows():
            tx.run("MERGE (t:Table {table_name: $name}) SET t.table_desc = $desc", 
                   name=row['Table Name'], desc=row['Table Description'])
        
        # Columns
        for _, row in df.iterrows():
            tx.run("""
            MERGE (c:Column {table_name: $tname, column_name: $cname})
            SET c.col_desc = $cdesc, c.fill_rate = $fr, c.sample_vals = $sv
            WITH c
            MATCH (t:Table {table_name: $tname})
            MERGE (c)-[:`column within table`]->(t)
            """, tname=row['Table Name'], cname=row['Column Name'], 
                cdesc=row['Column Description'], fr=row['Fill Rate'], sv=row['Unique Values'])

    def _run_semantic_matching_and_verification(self, df):
        """Runs the matching logic to create relationships."""

        
        if not self.chroma_client:

            return

        # Chroma Setup
        desc_coll = self.chroma_client.get_or_create_collection("descriptions", metadata={"hnsw:space": "cosine"})
        samp_coll = self.chroma_client.get_or_create_collection("samples", metadata={"hnsw:space": "cosine"})
        
        # Embedding loop (optimized to skip existing)
        existing_ids = set(desc_coll.get()['ids'])
        items_to_embed = [row for _, row in df.iterrows() if f"{row['Table Name']}.{row['Column Name']}" not in existing_ids]
        
        if items_to_embed:

            # Batch this if needed, simple loop for now
            for _, row in tqdm(pd.DataFrame(items_to_embed).iterrows(), total=len(items_to_embed)):
                col_id = f"{row['Table Name']}.{row['Column Name']}"
                desc_text = f"Column: {row['Column Name']}. Description: {row['Column Description']}"
                samp_text = f"Column: {row['Column Name']}. Samples: {row['Unique Values']}"
                
                d_emb = self._get_embedding_with_rotation(desc_text)
                s_emb = self._get_embedding_with_rotation(samp_text)
                
                if d_emb: desc_coll.add(ids=[col_id], embeddings=[d_emb], metadatas=[{"table": row['Table Name']}])
                if s_emb: samp_coll.add(ids=[col_id], embeddings=[s_emb], metadatas=[{"table": row['Table Name']}])

        # Matching Logic
        verified_results = []
        processed_pairs = set()
        
        all_desc = desc_coll.get(include=['embeddings'])
        cached_desc = {id: emb for id, emb in zip(all_desc['ids'], all_desc['embeddings'])}

        # Simplified matching loop (checking top 5 neighbors)

        for _, row_a in tqdm(df.iterrows(), total=len(df)):
            id_a = f"{row_a['Table Name']}.{row_a['Column Name']}"
            if id_a not in cached_desc: continue

            res = desc_coll.query(query_embeddings=[cached_desc[id_a]], n_results=5)
            
            for i in range(len(res['ids'][0])):
                id_b = res['ids'][0][i]
                sim_desc = 1 - res['distances'][0][i]
                
                if id_a == id_b: continue
                pair = tuple(sorted([id_a, id_b]))
                if pair in processed_pairs: continue
                processed_pairs.add(pair)
                
                t1, c1 = id_a.split('.')
                t2, c2 = id_b.split('.')
                if t1 == t2: continue 

                if sim_desc > 0.8:
                    # Verification logic
                    is_valid, _ = self._verify_join(t1, c1, t2, c2)
                    if is_valid:
                        verified_results.append({"source": id_a, "target": id_b, "score": sim_desc * 100})

        # Ingest Relationships
        if verified_results:

            with self.driver.session() as session:
                session.run("""
                UNWIND $batch AS row
                MATCH (c1:Column {table_name: split(row.source, '.')[0], column_name: split(row.source, '.')[1]})
                MATCH (c2:Column {table_name: split(row.target, '.')[0], column_name: split(row.target, '.')[1]})
                MERGE (c1)-[r:potentially_same_column]-(c2)
                SET r.confidence_score = row.score
                """, batch=verified_results)

    def _verify_join(self, t1, col1, t2, col2):
        """Verifies if two columns can actually join by checking local CSV files."""
        file1 = os.path.join(DATALAKE_PATH, f"{t1}.csv")
        file2 = os.path.join(DATALAKE_PATH, f"{t2}.csv")
        
        if not os.path.exists(file1) or not os.path.exists(file2):
            return False, 0
        
        try: 
            df1 = pd.read_csv(file1, usecols=[col1])
            df2 = pd.read_csv(file2, usecols=[col2])
            common = set(df1[col1].dropna().unique()).intersection(set(df2[col2].dropna().unique()))
            return len(common) > 0, len(common)
        except: return False, 0

    def _create_vector_index(self, df):
        """Enriches nodes with embeddings and creates index."""

        with self.driver.session() as session:
            # 1. Enrich Nodes
            for _, r in tqdm(df.iterrows(), total=len(df)):
                content = f"Table: {r['Table Name']}. Column: {r['Column Name']}. Description: {r['Column Description']}. Samples: {r['Unique Values']}"
                emb = self._get_embedding_with_rotation(content)
                if emb:
                    session.run("MATCH (c:Column {table_name: $t, column_name: $n}) SET c.embedding = $emb", 
                                t=r['Table Name'], n=r['Column Name'], emb=emb)
            
            # 2. Create Index
            try: session.run("DROP INDEX column_vector_index")
            except: pass
            
            # Use dummy embedding for dimension
            dummy = self._get_embedding_with_rotation("test")
            if dummy:
                session.run("""
                CREATE VECTOR INDEX column_vector_index FOR (c:Column) ON (c.embedding)
                OPTIONS {indexConfig: {`vector.dimensions`: $dim, `vector.similarity_function`: 'cosine'}}
                """, dim=len(dummy))

    @trace_tool("RAG Candidates")
    def retrieve_candidates_for_kpis(self, kpis):
        """
        Retrieves top candidate tables using GraphRAG (Neo4j Vector Search + Graph Traversal).
        """

        candidates_map = {}
        
        if not self.driver:

            return {}

        for kpiname in kpis:
            kpi_key = kpiname.split(":")[0].strip()
            embedding = self._get_embedding_with_rotation(kpiname)
            if not embedding or len(embedding) == 0: continue
            
            # Neo4j Vector Query
            cypher_query = """
            CALL db.index.vector.queryNodes('column_vector_index', 10, $emb) 
            YIELD node AS col, score
            MATCH (col)-[:`column within table`]->(t:Table)
            OPTIONAL MATCH (col)-[r:potentially_same_column]-(other:Column)
            RETURN col.column_name as name, col.col_desc as desc, 
                   t.table_name as table, t.table_desc as table_desc, score
            """
            
            try:
                with self.driver.session() as session:
                    results = session.run(cypher_query, emb=embedding)
                    kpi_candidates = []
                    
                    # Deduplicate tables, keep highest score
                    seen_tables = set()
                    
                    for i, res in enumerate(results):
                        table_name = res['table']
                        # Check table description for basic filtering? No, assume all in DB are valid.
                        if table_name in seen_tables: continue
                        seen_tables.add(table_name)
                        
                        full_schema = self._get_table_full_schema(table_name)
                        
                        kpi_candidates.append({
                            "Table": table_name,
                            "Similarity": res['score'],
                            "Source": "GraphRAG",
                            "Rank": i + 1,
                            "Full_Schema": full_schema 
                        })
                    
                    candidates_map[kpi_key] = kpi_candidates
            except Exception as e:
                pass
                
        return candidates_map

    def _get_table_full_schema(self, table_name):
        """Queries Neo4j to reconstruct the full schema text for a table."""
        try:
            with self.driver.session() as session:
                res = session.run("""
                MATCH (t:Table {table_name: $name})<-[:`column within table`]-(c:Column)
                RETURN t.table_desc as tdesc, c.column_name as cname, c.col_desc as cdesc
                """, name=table_name)
                
                records = list(res)
                if not records: return f"Table: {table_name}\n(No schema found)"
                
                desc = records[0]['tdesc'] or ""
                schema = f"Table: {table_name}\nDesc: {desc}\nColumns:\n"
                for r in records:
                    schema += f" - {r['cname']}: {r['cdesc']}\n"
                return schema
        except:
            return f"Table: {table_name}\n(Schema retrieval error)"

    @trace_tool("Metadata Identification")
    def identify_required_metadata(self, hypothesis, methodology, kpi_candidates):
        """
        Phase 4.0: Selection of Master Metadata Table (Grounding).
        Uses the detailed prompt structure as requested.
        """

        
        # Flatten schemas for prompt
        seen_tables = set()
        schemas_text = ""
        
        if not kpi_candidates:
            pass
        
        for kpi, candidates in kpi_candidates.items():
            for c in candidates:
                if c['Table'] not in seen_tables:
                    seen_tables.add(c['Table'])
                    schemas_text += c['Full_Schema'] + "\n" + "-"*20 + "\n"

        # The Detailed Prompt (as requested)
        prompt = f"""
        You are a Principal Data Architect. Based on the Hypothesis and Methodology, identify the MINIMUM set of tables and columns required to calculated the KPIs.
        
        Hypothesis: "{hypothesis}"
        Methodology:
        {methodology}
        
        Candidate Table Schemas (Retrieved via GraphRAG):
        {schemas_text}
        
        Task:
        Return a JSON nested list representing the 'Master Metadata Table'.
        Format:
        [
            ["S.No", "Table", "KPIs", "Columns", "Reasoning"],
            [1, "TABLE_A", "Churn Rate", "cust_id, churn_flag", "Primary fact table for churn."],
            ...
        ]
        
        Constraints:
        - Identify the minimum necessary tables to avoid data overhead.
        - Ensure all columns exist in the provided schemas.
        - Return ONLY the JSON list.
        """
        
        if gemini_client:
            # Use gemini_client.call_gemini if available
            try:
                response = gemini_client.call_gemini(prompt)
                clean_json = response.replace("```json", "").replace("```", "").strip()
                # Attempt to fix common JSON issues if simple load fails?
                try:
                    metadata_list = json.loads(clean_json)
                    return metadata_list
                except:
                     # Try to eval if it's a list string
                     try:
                         metadata_list = ast.literal_eval(clean_json)
                         if isinstance(metadata_list, list): return metadata_list
                     except:
                         pass

                     return []
            except Exception as e:

                return []
        else:

            return []

if __name__ == "__main__":
    retriever = RAGRetriever()

    kpis = ["Churn Rate"]
    candidates = retriever.retrieve_candidates_for_kpis(kpis)

