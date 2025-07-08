# Prompt Templates

This directory contains all prompt templates used by the Response Generation service. Prompts are externalized to allow easy modification without code changes.

## Structure

- `prompts.json` - Single consolidated file containing all prompts, organized by:
  - `base_system_prompt` - Base system prompt used for all queries
  - `intents` - All intent-specific prompts (system, user, guidelines, examples)
  - `special_prompts` - Follow-up and refinement prompts

## Supported Intents

1. **comparison** - Compare requirements across regulations
2. **specific_requirement** - Answer specific regulatory questions
3. **general_inquiry** - Provide general information about topics
4. **clarification** - Clarify concepts or requirements
5. **timeline** - Provide timeline/deadline information
6. **compliance_check** - Assess compliance requirements
7. **definition** - Define regulatory terms

## Modifying Prompts

1. Edit the relevant `.txt` or `.json` file
2. No code changes required
3. Changes take effect on next prompt template reload

## Template Variables

User prompt templates support these variables:
- `{query}` - User's query
- `{context}` - Retrieved regulatory context
- `{regulatory_bodies}` - List of regulations in context
- `{topics}` - Identified topics
- `{date_range}` - Date range of regulations
- `{conversation_history}` - Previous conversation (for clarifications)

## Adding New Intents

1. Add the new intent to the `intents` section in `prompts.json`:
   ```json
   "new_intent": {
     "system": "System prompt for the new intent...",
     "user": "User prompt template with {variables}...",
     "guidelines": [
       "Guideline 1",
       "Guideline 2"
     ],
     "example_format": "Optional example format...",
     "metadata_instructions": "Optional metadata instructions..."
   }
   ```

2. Update `QueryIntent` enum in `src/response_generation/prompt_templates.py` if needed

## Best Practices

- Keep prompts concise and clear
- Use specific instructions for GPT-4 model
- Include examples in guidelines when helpful
- Test prompts with various queries
- Version control prompt changes