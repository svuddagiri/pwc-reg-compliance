#!/usr/bin/env python3
"""
Generate bcrypt password hash for SQL script
"""
import bcrypt

def generate_password_hash(password: str) -> str:
    """Generate bcrypt hash for password"""
    salt = bcrypt.gensalt()
    hash_bytes = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hash_bytes.decode('utf-8')

if __name__ == "__main__":
    password = "TempPass123!"
    password_hash = generate_password_hash(password)
    
    print(f"Password: {password}")
    print(f"Hash: {password_hash}")
    print("\nReplace @password_hash in the SQL script with:")
    print(f"DECLARE @password_hash NVARCHAR(255) = '{password_hash}';")
    
    # Verify the hash works
    is_valid = bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    print(f"\nHash verification: {'✓ Valid' if is_valid else '✗ Invalid'}")