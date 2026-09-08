---
name: SIH1
description: Describe what this custom agent does and when to use it.
argument-hint: The inputs this agent expects, e.g., "a task to implement" or "a question to answer".
# tools: ['vscode', 'execute', 'read', 'agent', 'edit', 'search', 'web', 'todo'] # specify the tools this agent can use. If not set, all enabled tools are allowed.
---

<!-- Tip: Use /create-agent in chat to generate content with agent assistance -->

Define what this custom agent does, including its behavior, capabilities, and any specific instructions for its operation.I want you to route my LLM calls for "gpt-6-astra" through the Experiential gateway instead of
calling the provider directly. It speaks the OpenAI Chat Completions API, so this is a base-URL
and key swap. Please:

1. Point the client at https://api.experientiallabs.ai/v1 as the base URL.
2. Authenticate with my Experiential API key from the EXPLABS_API_KEY environment variable. If
   it isn't set, stop and tell me to create one under Settings -> API keys and export it.
3. Use the model id "gpt-6-astra" exactly.
4. Update every place my code builds an LLM client for this model to use that base URL and key,
   leaving streaming and tool-calls as they are.
5. Make one test call and show me the reply plus the token usage, so we confirm it runs on my
   Experiential credits.

Tell me which files you changed.
