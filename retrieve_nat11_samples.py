#!/usr/bin/env python3
import sys
import os
sys.path.append('.')

import lancedb
import json
from rag_system.indexing.embedders import LanceDBManager

def get_sample_chunks(index_name="nat11", limit=5):
    # Based on the database query, the collection name is text_pages_1f90d332-1f94-4c63-8472-23380c86e025
    collection_name = "text_pages_1f90d332-1f94-4c63-8472-23380c86e025"
    
    DB_PATH = "./rag_system/index_store/lancedb"
    
    try:
        db_manager = LanceDBManager(db_path=DB_PATH)
        table = db_manager.get_table(collection_name)
        
        print(f"Retrieving {limit} sample chunks from index '{index_name}'...")
        print("=" * 60)
        
        # Get sample data
        df = table.limit(limit).to_pandas()
        
        for i, row in df.iterrows():
            print(f"\n--- Chunk {i+1} ---")
            print(f"Chunk ID: {row.get('chunk_id', 'N/A')}")
            print(f"Document ID: {row.get('document_id', 'N/A')}")
            print(f"Chunk Index: {row.get('chunk_index', 'N/A')}")
            
            # Parse metadata if it exists
            metadata_str = row.get('metadata', '{}')
            try:
                metadata = json.loads(metadata_str)
                original_text = metadata.get('original_text', row.get('text', 'N/A'))
                print(f"Original Text (first 200 chars): {original_text[:200]}...")
                if len(original_text) > 200:
                    print(f"Full text length: {len(original_text)} characters")
                
                # Show other metadata
                for key, value in metadata.items():
                    if key not in ['original_text', 'text'] and not key.startswith('_'):
                        if isinstance(value, str) and len(value) > 100:
                            print(f"{key}: {value[:100]}... (truncated)")
                        else:
                            print(f"{key}: {value}")
                            
            except json.JSONDecodeError:
                print(f"Text: {row.get('text', 'N/A')}")
            
            print("-" * 40)
            
    except Exception as e:
        print(f"Error retrieving chunks: {e}")
        print(f"Available tables: {db_manager.db.table_names()}")

if __name__ == "__main__":
    get_sample_chunks()