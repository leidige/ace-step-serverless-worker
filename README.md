# ACE-Step 1.5 RunPod Serverless Worker (Network Volume models)

## What this is
- Lightweight Docker image: ACE code + RunPod handler only
- Models download to `/runpod-volume/models/acestep` on first `setup` / `cover` job
- Use with Network Volume `ace-models-50gb` (US-CA-2)

## Actions
| action | purpose |
|--------|---------|
| `health` | Check volume + CUDA |
| `setup` / `download` | Download turbo + vae + embedding + LM 0.6B from HF |
| `cover` | Remix/Cover from `reference_audio_base64` |

## Deploy
1. Push this folder to GitHub
2. RunPod Console → Serverless → New Endpoint → Deploy from GitHub
3. Attach Network Volume `ace-models-50gb`
4. workersMin=0, idleTimeout=60–120, GPU 24GB
5. First request: `{"input":{"action":"setup"}}`
6. Cover test: `{"input":{"action":"cover","reference_audio_base64":"...","audio_duration":30}}`

Do not attach to mo-mirror-fast.
