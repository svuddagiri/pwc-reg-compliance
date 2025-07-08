"""
Check if our tables exist in the database
"""
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.clients.azure_sql import AzureSQLClient


async def check_tables():
    """Check which of our tables exist"""
    our_tables = [
        'users',
        'sessions', 
        'conversations',
        'messages',
        'search_queries',
        'user_preferences'
    ]
    
    async with AzureSQLClient() as sql_client:
        # Get all tables
        all_tables = await sql_client.execute_query("""
            SELECT TABLE_NAME 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_TYPE = 'BASE TABLE'
            ORDER BY TABLE_NAME
        """)
        
        existing_tables = [t['TABLE_NAME'] for t in all_tables]
        
        print("Checking for our tables:")
        print("-" * 40)
        
        for table in our_tables:
            if table in existing_tables:
                print(f"✅ {table} - EXISTS")
            else:
                print(f"❌ {table} - NOT FOUND")
        
        # Check if we have any users
        if 'users' in existing_tables:
            users = await sql_client.execute_query("SELECT COUNT(*) as count FROM users")
            print(f"\nUsers table has {users[0]['count']} records")
        
        return all(table in existing_tables for table in our_tables)


if __name__ == "__main__":
    all_exist = asyncio.run(check_tables())