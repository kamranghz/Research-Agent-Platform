# Model Context Protocol (MCP)

## What is MCP?

The Model Context Protocol (MCP) is an open standard for connecting AI applications to external systems.

Using MCP, AI applications such as Claude, ChatGPT, IDEs, and other AI agents can connect to:

- Data sources (local files, databases, APIs)
- Tools (calculators, search engines, browsers)
- Workflows and reusable prompts

A useful analogy is that MCP is like a USB-C port for AI applications. Just as USB-C provides a standardized interface for connecting hardware devices, MCP provides a standardized interface for connecting AI applications to external systems. :contentReference[oaicite:0]{index=0}

---

# Why MCP Matters

Without MCP, every AI application would need to build custom integrations for every external system.

MCP provides a common protocol that allows developers to build integrations once and reuse them across different AI applications.

Benefits include:

- Reduced development complexity
- Reusable integrations
- Better interoperability
- More capable AI agents
- Easier integration with external services

---

# MCP Architecture

MCP follows a client-server architecture.

The main participants are:

- **MCP Host**
  - The AI application (for example Claude Desktop, Visual Studio Code, or another AI application).

- **MCP Client**
  - A protocol component created by the host.
  - Each client maintains a dedicated connection to exactly one MCP server.

- **MCP Server**
  - A program that exposes capabilities to AI applications.

An AI application may connect to multiple MCP servers simultaneously. :contentReference[oaicite:1]{index=1}

---

# MCP Layers

MCP consists of two layers.

## 1. Data Layer

The Data Layer defines the JSON-RPC based protocol used between MCP clients and servers.

It includes:

- Capability discovery
- Tool execution
- Resources
- Prompts
- Notifications

## 2. Transport Layer

The Transport Layer defines how communication occurs.

Supported transports include:

- STDIO
- Streamable HTTP

The transport layer handles:

- Connection establishment
- Authentication
- Message framing
- Secure communication

:contentReference[oaicite:2]{index=2}

---

# Core MCP Primitives

MCP defines three core primitives.

## Tools

Tools are executable functions.

Examples:

- Read files
- Call APIs
- Query databases
- Search the web

The AI model decides when to invoke tools. :contentReference[oaicite:3]{index=3}

---

## Resources

Resources provide contextual information.

Examples:

- File contents
- Database records
- API responses
- Knowledge bases

Resources are passive data sources that applications retrieve and provide as context to the model. :contentReference[oaicite:4]{index=4}

---

## Prompts

Prompts are reusable templates.

Examples:

- Plan a vacation
- Summarize meetings
- Draft an email

Prompts provide structured interaction patterns and are explicitly selected by users. :contentReference[oaicite:5]{index=5}

---

# Typical MCP Workflow

A typical MCP interaction follows these steps:

1. Client discovers server capabilities.
2. Client lists available tools.
3. The language model decides to call a tool.
4. The client executes the tool on the MCP server.
5. The tool result is returned.
6. The language model continues reasoning using the returned result.

This discovery → tool selection → tool execution workflow is one of the core concepts of MCP. :contentReference[oaicite:6]{index=6}

---

# MCP and AI Agents

MCP enables AI agents to:

- Access external tools
- Read external data
- Interact with APIs
- Connect to multiple systems through one standard protocol

Instead of implementing separate integrations for every service, AI applications communicate through the MCP protocol.

---

# MCP in This Project

This Research Agent Platform currently implements native LangGraph tools.

The project does not yet include an MCP client or MCP server.

Future versions may expose tools such as:

- File Reader
- Retriever
- Browser
- API tools

through an MCP server.