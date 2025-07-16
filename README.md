# Dumb LLM Router

An intelligent routing system for multiple AI providers that automatically selects the best model based on request complexity, research needs, and context. Perfect for OpenWebUI integration.

## 🚀 Features

- **Intelligent Model Selection**: Automatically routes requests to the most appropriate AI model
- **Multi-Provider Support**: OpenAI, Google Gemini, Anthropic Claude, Perplexity
- **Research Detection**: Automatically detects when web search/research is needed
- **Vision Support**: Handles image analysis requests with vision-capable models
- **Context Awareness**: Manages conversation context and model stickiness
- **OpenWebUI Integration**: Seamless integration with OpenWebUI as a single endpoint
- **Secure**: Runs as non-root user with security best practices
- **Unraid Ready**: Pre-configured for Unraid with template included

## 🧠 How It Works

The Smart LLM Router intelligently classifies incoming requests and routes them to the optimal model:

### Request Classification
- **Simple Requests** → Fast, cost-effective models (GPT-4.1-nano, Gemini Flash)
- **Complex Requests** → Advanced models (Gemini Pro, GPT-4o)
- **Research Requests** → Research-enabled models (Perplexity, Gemini with Search)
- **Vision Requests** → Multimodal models (Gemini Pro with vision)
- **Escalation Requests** → Premium models for challenging tasks

### Model Routing Examples
```
"What's 2+2?" → GPT-4.1-nano (simple, fast)
"Research the latest AI developments" → Perplexity (research)
"Analyze this image" → Gemini Pro (vision)
"Write a complex analysis of..." → Gemini Pro (complex)
"Escalate this request" → Premium model
```

## 📋 Requirements

- Docker & Docker Compose
- API keys for desired providers:
  - OpenAI API key (required)
  - Google Gemini API key (required)  
  - Anthropic API key (optional)
  - Perplexity API key (optional)

## 🐳 Quick Start

### Option 1: Docker Compose (Recommended)

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/dumb-llm-router.git
cd dumb-llm-router
```

2. **Create environment file**:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. **Start the router**:
```bash
docker-compose up -d
```

### Option 2: Unraid Template

1. **Add template URL** in Unraid Apps:
```
https://raw.githubusercontent.com/your-username/dumb-llm-router/main/unraid-template.xml
```

2. **Configure required settings**:
   - OpenAI API Key
   - Gemini API Key
   - Host Port (default: 8005)
   - Config directory path

3. **Apply and start**

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `OPENAI_API_KEY` | Yes | OpenAI API key | - |
| `GEMINI_API_KEY` | Yes | Google Gemini API key | - |
| `ANTHROPIC_API_KEY` | No | Anthropic Claude API key | - |
| `PERPLEXITY_API_KEY` | No | Perplexity API key | - |
| `OLLAMA_API_HOST` | No | Ollama endpoint for local models | - |
| `CONFIG_PATH` | No | Configuration directory | `/config` |
| `LOG_LEVEL` | No | Logging level | `INFO` |

### Model Configuration

Models are configured in `config/models.json`:

```json
{
  "models": {
    "classifier": {
      "provider": "openai",
      "model_id": "gpt-4.1-nano",
      "type": "chat"
    },
    "4.1-nano": {
      "provider": "openai", 
      "model_id": "gpt-4.1-nano",
      "type": "chat"
    }
  }
}
```

### Routing Configuration

Routing logic is configured in `config/config.yaml`:

```yaml
routing:
  classifier_model: classifier
  simple_no_research_model: 4.1-nano
  simple_research_model: 4o-mini
  hard_no_research_model: Flash-No-Research
  hard_research_model: Flash-Research
  escalation_model: Gemini-Pro
  fallback_model: Flash-No-Research

context_detection:
  token_usage_threshold: 4000
  
logging:
  level: INFO
  enable_detailed_routing_logs: true
```

## 🔌 OpenWebUI Integration

### Setup Steps

1. **Start Smart LLM Router** (port 8005)

2. **Add to OpenWebUI**:
   - Go to **Settings** → **Models**  
   - Add **OpenAI API**:
     - Base URL: `http://your-server-ip:8005/v1`
     - API Key: `any-key-works`
   - The router will appear as `dumb-llm-router`

3. **Configure Image Generation**:
   - Go to **Admin Panel** → **Settings** → **Images**
   - Set Engine to **Gemini**
   - API Base URL: `https://generativelanguage.googleapis.com/v1beta`
   - API Key: Your Gemini API key
   - Model: `imagen-4.0-generate-preview-06-06`

### Usage in OpenWebUI

- **Chat**: Select `dumb-llm-router` as your model
- **Images**: Use the 🖼️ button for image generation
- **Research**: Router automatically detects and handles research requests
- **Vision**: Upload images and ask questions about them

## 📊 API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /v1/models` | List available models |
| `POST /v1/chat/completions` | Main chat endpoint |
| `GET /health` | Health check |
| `GET /debug/conversations` | Debug conversation states |

### Example Request

```bash
curl -X POST http://localhost:8005/v1/chat/completions \
  -H "Authorization: Bearer any-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dumb-llm-router",
    "messages": [
      {"role": "user", "content": "Research the latest developments in AI"}
    ]
  }'
```

## 🔧 Development

### Local Development

1. **Clone and setup**:
```bash
git clone https://github.com/your-username/dumb-llm-router.git
cd dumb-llm-router
pip install -r requirements.txt
```

2. **Configure environment**:
```bash
cp .env.example .env
# Add your API keys
```

3. **Run locally**:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Building Docker Image

```bash
docker build -t dumb-llm-router .
docker run -p 8005:8000 --env-file .env dumb-llm-router
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   OpenWebUI     │───▶│ Dumb LLM Router  │───▶│   AI Providers  │
│                 │    │                  │    │                 │
│ - Chat UI       │    │ - Classification │    │ - OpenAI        │
│ - Image Gen     │    │ - Model Selection│    │ - Gemini        │
│ - File Upload   │    │ - Context Mgmt   │    │ - Anthropic     │
└─────────────────┘    │ - Response Format│    │ - Perplexity    │
                       └──────────────────┘    └─────────────────┘
```

### Key Components

- **Classifier**: Analyzes request complexity and requirements
- **Router**: Selects optimal model based on classification
- **Clients**: Handles API communication with each provider
- **State Manager**: Manages conversation context and stickiness

## 🛠️ Troubleshooting

### Common Issues

**Container won't start:**
```bash
# Check logs
docker logs dumb-llm-router

# Verify API keys are set
docker exec dumb-llm-router env | grep API_KEY
```

**OpenWebUI can't connect:**
- Verify port 8005 is accessible
- Check Docker network settings
- Ensure API endpoint is correct: `http://ip:8005/v1`

**Models not responding:**
- Check API keys are valid
- Verify internet connectivity
- Review model configuration in `config/models.json`

### Health Check

```bash
curl http://localhost:8005/health
# Should return: {"status": "ok", "message": "Smart LLM Router v2.0 is running"}
```

### Debug Mode

Set `LOG_LEVEL=DEBUG` to see detailed routing decisions:

```bash
docker-compose down
echo "LOG_LEVEL=DEBUG" >> .env
docker-compose up -d
docker logs -f dumb-llm-router
```

## 🔒 Security

- Runs as non-root user (UID 99, GID 100 for Unraid)
- Read-only filesystem with minimal writable areas
- Dropped Linux capabilities 
- No new privileges allowed
- Resource limits enforced
- API keys stored securely in environment variables

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/dumb-llm-router/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/dumb-llm-router/discussions)

## 🙏 Acknowledgments

- [OpenWebUI](https://github.com/open-webui/open-webui) for the excellent AI interface
- [Unraid](https://unraid.net/) for the containerization platform
- All the AI providers for their APIs

---

**Made with ❤️ for the Unraid and OpenWebUI communities**
