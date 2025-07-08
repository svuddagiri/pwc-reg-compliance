# Vector DB - Consent Chunks Retrieval

This folder contains tools for retrieving and analyzing consent chunks from Azure AI Search.

## Overview

The main script `retrieve_consent_chunks.py` retrieves the 41 consent chunks from the Azure AI Search index using the consent filter configuration defined in `/config/filters/consent_filters.json`.

## Usage

```bash
# Retrieve consent chunks and save to CSV
python retrieve_consent_chunks.py
```

## Output Files

The script generates two files:

1. **consent_chunks_YYYYMMDD_HHMMSS.csv** - Contains all consent chunks with the following fields:
   - chunk_id
   - chunk_text
   - clause_type
   - clause_domain
   - clause_subdomain
   - jurisdiction
   - regulation
   - section
   - keywords
   - entities
   - has_embedding

2. **consent_chunks_YYYYMMDD_HHMMSS.summary.txt** - Summary statistics including:
   - Total number of chunks
   - Unique jurisdictions
   - Unique regulations
   - Chunks per jurisdiction

## Filter Configuration

The consent filter uses:
- Domain: "Individual Rights Processing"
- Subdomain: "Consent"
- Expected results: 41 chunks from 9 jurisdictions

## Expected Jurisdictions

Based on the configuration, we expect chunks from:
- European Union (GDPR)
- United States (various state laws)
- Brazil (LGPD)
- Costa Rica
- Denmark
- Estonia
- Gabon
- Quebec
- And others