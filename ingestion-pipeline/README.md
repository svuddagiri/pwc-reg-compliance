# Regulatory Document Processing Pipeline

A standalone document processing pipeline for extracting, enriching, and indexing regulatory documents with Azure AI services.

## 📁 Project Structure

```
clean-ingestion/
├── config/              # Configuration management
├── core/                # Abstract base classes
├── database/            # Database implementations
├── handlers/            # Document processors
├── scripts/             # CLI scripts
├── utils/               # Shared utilities
├── *.sh                 # Shell scripts for operations
└── setup.py             # Package installation
```

See [FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md) for detailed structure.

## Features

- **Document Processing**: Extract metadata, structure, and content from regulatory PDFs
- **Intelligent Chunking**: Advanced clause-based chunking with hierarchy preservation
- **AI Enrichment**: Generate summaries and extract regulatory entities using Azure OpenAI
- **Multi-Index Storage**: Store processed documents in Azure AI Search with vector embeddings
- **Regulation Normalization**: Standardize regulation names and extract official titles
- **Entity Recognition**: Extract actors, data types, penalties, obligations, and rights
- **Clause Classification**: Automatically classify clause types and domains
- **No API Server Required**: Direct pipeline execution through CLI or standalone scripts

## Prerequisites

1. Python 3.8 or higher
2. Azure subscription with the following services configured:
   - Azure Storage Account (for document storage)
   - Azure Document Intelligence (for PDF structure extraction)
   - Azure OpenAI (for AI enrichment and classification)
   - Azure AI Search (for document indexing and retrieval)
   - SQL Server (local or Azure SQL Database)

## Cross-Platform Support

The pipeline supports Windows, Linux, and macOS:

### Shell Scripts (Linux/Mac)
```bash
./process_documents.sh --batch-size 10
./monitor_pipeline.sh --watch
./reset_database.sh
```

### Batch Files (Windows)
```batch
process_documents.bat --batch-size 10
monitor_pipeline.bat
reset_database.bat
```

### Python CLI (All Platforms)
```bash
python pipeline process --preset full
python pipeline monitor --once
python pipeline reset --status
```

## Installation

1. Clone the repository and navigate to the clean-ingestion folder:
```bash
# Windows
cd path\to\regulatory_chat_bot-2\clean-ingestion

# Linux/Mac
cd path/to/regulatory_chat_bot-2/clean-ingestion
```

2. Create a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

Edit the `.env` file with your Azure credentials:
```
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com/
AZURE_OPENAI_KEY=your-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large

# Azure Search
AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_SEARCH_KEY=your-key
AZURE_SEARCH_INDEX_NAME=regulatory-documents

# Azure Storage
AZURE_STORAGE_CONNECTION_STRING=your-connection-string
AZURE_STORAGE_CONTAINER_NAME=documents

# Azure Document Intelligence
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-doc-intelligence.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=your-key

# SQL Database (for local SQL Server on Windows)
SQL_SERVER=localhost
SQL_DATABASE=regulatory_compliance
SQL_USERNAME=sa
SQL_PASSWORD=YourStrong@Passw0rd

# For Azure SQL Database, use:
# SQL_USE_AZURE=true
# SQL_SERVER=your-server.database.windows.net
# SQL_DATABASE=regulatory_compliance
# SQL_USERNAME=your-username
# SQL_PASSWORD=your-password
```

## Usage

### Method 1: Pipeline CLI (Recommended)

The Pipeline CLI provides a unified interface for all operations:

```bash
# Windows
python pipeline_cli.py --help

# Linux/Mac
python pipeline_cli.py --help
```

#### Process Documents
```bash
# Process all PDFs in the container (quiet mode - only shows progress)
python pipeline_cli.py process

# Process with verbose logging (shows all details)
python pipeline_cli.py process --verbose

# Save logs to file
python pipeline_cli.py process --log-file processing.log

# Process specific documents
python pipeline_cli.py process --documents "document1.pdf" "document2.pdf"

# Process with custom configuration
python pipeline_cli.py process --config gdpr --max-workers 3

# Process with specific preset (fast, minimal, storage_only)
python pipeline_cli.py process --preset fast
```

#### Monitor Pipeline
```bash
# Monitor current pipeline status
python pipeline_cli.py monitor

# Monitor with auto-refresh every 5 seconds
python pipeline_cli.py monitor --refresh 5

# Show detailed status
python pipeline_cli.py monitor --detailed

# One-time status check
python pipeline_cli.py monitor --once
```

#### Reset Database
```bash
# Interactive reset
python pipeline_cli.py reset

# Reset specific table groups
python pipeline_cli.py reset --group pipeline --force

# Reset all tables except users
python pipeline_cli.py reset --group all --force

# View database status only
python pipeline_cli.py reset --status
```

#### List Configurations
```bash
# Show available pipeline configurations
python pipeline_cli.py list-configs

# Create custom configuration
python pipeline_cli.py config --create my_config.yaml
```

### Method 2: Standalone Scripts

You can also run individual scripts directly:

#### Process Documents
```bash
# Windows
python process_documents.py

# With options
python process_documents.py --blob-prefix "gdpr" --max-workers 5 --reprocess

# With verbose logging
python process_documents.py --verbose

# Save logs to file
python process_documents.py --log-file processing.log

# Linux/Mac
python process_documents.py
```

Options:
- `--blob-prefix`: Filter documents by prefix
- `--specific-blobs`: Process specific documents (comma-separated)
- `--max-workers`: Number of parallel workers (default: 3)
- `--reprocess`: Reprocess already processed documents
- `--preset`: Use configuration preset (full, fast, minimal, storage_only)
- `--verbose`: Show detailed logs (default: quiet mode)
- `--log-file`: Write logs to specified file

#### Monitor Pipeline
```bash
# Windows
python monitor_pipeline.py

# With auto-refresh
python monitor_pipeline.py --interval 10

# Linux/Mac
python monitor_pipeline.py
```

#### Reset Database
```bash
# Windows
python reset_database_standalone.py

# Linux/Mac
python reset_database_standalone.py
```

Options:
- `--tables`: Specific tables to reset
- `--group`: Reset a group of tables (pipeline, storage, monitoring, auth, all)
- `--status`: Show database status only
- `--force`: Skip confirmation prompt

## Pipeline Architecture

### 1. Document Ingestion
- Downloads PDFs from Azure Blob Storage
- Extracts document structure using Azure Document Intelligence
- Preserves page numbers, sections, tables, and hierarchies

### 2. Metadata Extraction
- Extracts comprehensive regulatory metadata:
  - Document type, framework, and jurisdiction
  - Regulation normalization (standardized keys)
  - Official regulation names and aliases
  - Temporal information (effective dates, deadlines)
  - Penalties and enforcement authorities
  - Cross-references and related documents

### 3. Intelligent Chunking
- Splits documents into semantic chunks (500-1000 tokens)
- Preserves clause boundaries and hierarchical structure
- Classifies clause types using Azure OpenAI:
  - Definitions, obligations, rights, penalties
  - Procedures, exceptions, notifications
  - Governance, compliance, security requirements

### 4. Entity Recognition
- Extracts regulatory entities:
  - Actors (data controllers, processors, subjects)
  - Data types (personal data, sensitive data)
  - Time periods and deadlines
  - Monetary amounts and penalties
  - Cross-references

### 5. AI Enrichment
- Generates summaries for each chunk
- Extracts clause domains and subdomains:
  - Individual Rights Processing
  - Data Lifecycle Management
  - Information Security
  - Third-Party Risk Management
  - And more...

### 6. Indexing
- Stores enriched chunks in Azure AI Search
- Creates vector embeddings for semantic search
- Enables hybrid search (keyword + vector)
- Supports faceted filtering by regulation, jurisdiction, clause type

## Configuration System

### Available Presets
- `full`: Complete processing with all features enabled
- `fast`: Faster processing without some AI enrichment
- `minimal`: Basic processing, SQL storage only
- `storage_only`: Focus on storage without AI features

### Custom Configuration
Create a custom configuration file:
```yaml
batch_size: 5
max_workers: 3
chunk_size: 750
chunk_overlap: 50
enable_embeddings: true
enable_ner: true
enable_azure_search: true
storage_backends:
  - sql
  - azure_search
```

## Database Schema

The pipeline uses SQL Server to track:
- **pipeline_jobs**: Processing job status and progress
- **document_status**: Individual document processing status
- **clauses**: Extracted regulatory clauses
- **query_history**: Search queries (for analytics)
- **audit_logs**: System audit trail
- **users**: User accounts (if authentication is enabled)

## Project Structure

```
clean-ingestion/
├── pipeline_cli.py              # Main CLI interface
├── run_pipeline.py              # Core pipeline runner
├── process_documents.py         # Document processor
├── monitor_pipeline.py          # Pipeline monitor
├── reset_database_standalone.py # Database manager
├── pipeline_config.py           # Configuration system
├── requirements.txt             # Python dependencies
├── config/
│   └── config.py               # Settings configuration
└── src/
    ├── ingestion/
    │   ├── blob_client.py               # Azure Blob Storage client
    │   ├── enhanced_document_processor.py # Main document processor
    │   ├── enhanced_pipeline_orchestrator.py # Pipeline orchestrator
    │   ├── metadata_extractor.py        # Metadata extraction
    │   ├── clause_chunker.py            # Basic chunking
    │   ├── advanced_clause_chunker.py   # Advanced chunking
    │   ├── enhanced_metadata_extractor.py # Enhanced metadata
    │   └── regulatory_entity_recognizer.py # Entity recognition
    ├── storage/
    │   ├── sql_database.py      # SQL database operations
    │   ├── azure_search.py      # Azure Search integration
    │   └── cosmos_db.py         # Cosmos DB integration
    ├── services/
    │   └── ai_enrichment_service.py # AI enrichment
    └── models/
        ├── embeddings.py        # Embedding generation
        └── citation_formatter.py # Citation formatting
```

## Monitoring

The pipeline provides real-time monitoring:
- Job progress and status
- Document processing statistics
- Error tracking and diagnostics
- Processing performance metrics
- Rich terminal UI with progress bars

## Troubleshooting

### Common Issues

1. **Connection Errors**
   - Verify all Azure credentials in `.env`
   - Check firewall settings for SQL Server
   - Ensure Azure services are accessible

2. **Processing Failures**
   - Check logs in the console output
   - Verify PDF files are valid and accessible
   - Ensure sufficient Azure OpenAI quota

3. **Empty Regulation Fields**
   - Verify Azure OpenAI is properly configured
   - Check that regulations are being detected in documents
   - Review logs for "regulation normalization" messages

### Windows-Specific Notes

1. **SQL Server Setup**
   - Install SQL Server Express (free)
   - Enable SQL Server Authentication
   - Create a database named `regulatory_compliance`
   - Update connection string in `.env`

2. **Python Path Issues**
   - Use full paths in commands if needed
   - Ensure Python is in your PATH
   - Use `python` instead of `python3` on Windows

3. **File Paths**
   - Use backslashes or raw strings for Windows paths
   - Example: `r"C:\path\to\file"` or `"C:\\path\\to\\file"`

## Performance Optimization

- **Parallel Processing**: Adjust `--max-workers` based on your system
- **Batch Size**: Configure batch size for optimal throughput
- **Caching**: Azure OpenAI responses are cached to avoid redundant calls
- **Indexing**: Documents are indexed in batches of 100 for efficiency
- **Presets**: Use `fast` preset for quicker processing without full AI enrichment

## Next Steps

After processing documents, you can:
1. Search documents using Azure AI Search
2. Build a chatbot interface using the indexed content
3. Create compliance dashboards and analytics
4. Export processed data for further analysis
5. Integrate with the main regulatory chatbot API

## Support

For issues or questions:
1. Check the logs for detailed error messages
2. Verify all Azure services are properly configured
3. Ensure documents are valid PDFs with text content
4. Review the troubleshooting section above
5. Use `--debug` flag for verbose logging