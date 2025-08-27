#!/usr/bin/env python3
import sys
import os
sys.path.append('.')

import lancedb
from rag_system.indexing.embedders import LanceDBManager

def check_nat11_embeddings():
    # The collection name from database query
    collection_name = "text_pages_1f90d332-1f94-4c63-8472-23380c86e025"
    
    # The correct path based on config analysis
    DB_PATH = "./index_store/lancedb"
    
    try:
        print(f"Connecting to LanceDB at: {DB_PATH}")
        db_manager = LanceDBManager(db_path=DB_PATH)
        
        print("Available tables:")
        table_names = db_manager.db.table_names()
        for table in table_names:
            print(f"  - {table}")
        
        if collection_name in table_names:
            table = db_manager.get_table(collection_name)
            df = table.limit(5).to_pandas()
            print(f"\nFound table '{collection_name}' with {len(df)} sample rows:")
            
            for i, row in df.iterrows():
                print(f"\n--- Row {i+1} ---")
                print(f"Chunk ID: {row.get('chunk_id', 'N/A')}")
                print(f"Document ID: {row.get('document_id', 'N/A')}")
                print(f"Text preview: {str(row.get('text', 'N/A'))[:100]}...")
                
        else:
            print(f"\nTable '{collection_name}' not found in LanceDB")
            print("This suggests the embeddings were never created or are in a different location")
            
    except Exception as e:
        print(f"Error checking embeddings: {e}")

if __name__ == "__main__":
    check_nat11_embeddings()