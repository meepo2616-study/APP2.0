# Field Breach Rapid Assessment Prototype

This Streamlit app is a preliminary field-breach assessment prototype.

It combines:

- rule-based breach mechanism stage judgment;
- simplified breach geometry and discharge process simulation;
- PE-XGBoost peak discharge prediction;
- physics-guided interaction factors.

## Files required for deployment

- `app.py`
- `PE_XGBoost.pkl`
- `pe_selected_features.pkl`
- `requirements.txt`

## Streamlit Community Cloud

1. Upload this folder to a GitHub repository.
2. Create a new Streamlit app from that repository.
3. Set the main file path to:

```text
app.py
```

The app uses the local model file `PE_XGBoost.pkl`; no external API key is required.

## Notes

The process simulation module is a simplified prototype. It is intended for rapid field screening and thesis demonstration. Full breach evolution and downstream inundation analysis should be coupled with the Chapter 3 numerical model and/or a 2D hydrodynamic model.
