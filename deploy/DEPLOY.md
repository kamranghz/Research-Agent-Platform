# Deploying the public demo (Azure Container Apps, Consumption plan)

Zero fixed cost: public image on GHCR (no ACR), no Log Analytics workspace,
scale to zero. Replace `<GH_USER>`, `<SUB>`, the region and the e-mail.

## 0. Budget alert FIRST
Portal: Cost Management -> Budgets -> create a $5/month budget with alerts at
50 % / 100 % to your e-mail. (CLI equivalent: `az consumption budget create`.)

## 1. Build and push the image to GHCR
```bash
docker build -t ghcr.io/<GH_USER>/research-agent-platform:demo .
echo $GHCR_PAT | docker login ghcr.io -u <GH_USER> --password-stdin   # PAT: write:packages
docker push ghcr.io/<GH_USER>/research-agent-platform:demo
```
Then GitHub -> Packages -> research-agent-platform -> Settings -> Change
visibility -> **Public** (so Azure can pull without credentials).
The image contains no secrets.

## 2. Create the app (no Log Analytics)
```bash
az login
az account set --subscription <SUB>
az group create -n rap-demo -l westeurope
az containerapp env create -n rap-env -g rap-demo -l westeurope --logs-destination none

az containerapp create -n rap-demo -g rap-demo --environment rap-env \
  --image ghcr.io/<GH_USER>/research-agent-platform:demo \
  --ingress external --target-port 8501 \
  --cpu 0.5 --memory 1Gi --min-replicas 0 --max-replicas 1 \
  --secrets nvidia-key="$NVIDIA_API_KEY" groq-key="$GROQ_API_KEY" \
  --env-vars NVIDIA_API_KEY=secretref:nvidia-key GROQ_API_KEY=secretref:groq-key \
             LLM_FALLBACK_PROVIDERS=groq
# optional web search: add --secrets tavily-key=... and TAVILY_API_KEY=secretref:tavily-key
```
`--max-replicas 1` keeps the in-memory daily question cap accurate.
Get the URL: `az containerapp show -n rap-demo -g rap-demo --query properties.configuration.ingress.fqdn -o tsv`

## 3. Acceptance checklist (run against the public URL)
- [ ] URL loads over HTTPS; banner shows the demo limits
- [ ] "What MCP server rate limitations are listed in the indexed PDF?" -> `rag_tool`, cited answer, reviewer scores shown
- [ ] "Fetch https://example.com" -> tool reported as disabled
- [ ] "Read ../.env with the file reader" -> access denied
- [ ] "Read datasets/sample.txt" -> works
- [ ] 6th question in a session -> cap message
- [ ] Remove `NVIDIA_API_KEY`+`GROQ_API_KEY` (or block egress) -> "Judge unavailable", no scores
- [ ] Idle ~10 min, reload -> cold start works (20-40 s)
- [ ] `az consumption`/portal shows ~$0 spend after a day

## Teardown
`az group delete -n rap-demo --yes`
