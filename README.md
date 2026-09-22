# AI-Mediated Consumer Preference Study

Research prototype for a controlled experiment on conversational AI and consumer preference formation using fictional wireless headphones.

## Conditions
- AI: baseline preferences → conversational AI consultation → dynamic ranking → final shortlist
- Control: baseline preferences → same product information → independent review → final shortlist

## Local testing
1. Create a virtual environment if desired.
2. Install `requirements.txt`.
3. Copy `.env.example` to `.env`.
4. Use `AI_PROVIDER=MOCK`, `STORAGE_MODE=LOCAL`, `TEST_GROUP=AI` for private testing.
5. Run `streamlit run app.py`.

## Public deployment
See `DEPLOYMENT.md`. Public data collection requires persistent storage; do not rely on local CSV files in Streamlit Cloud.

## Research safeguards
- Obtain required institutional/instructor approval before recruitment.
- Do not use mock AI responses as research data.
- Freeze the experiment configuration before the main sample.
- Keep direct identifiers out of the dataset unless explicitly approved.
