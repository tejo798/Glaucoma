# Glaucoma
👁️ Glaucoma Risk Prediction

This project predicts the risk of Glaucoma based on patient health metrics using a machine learning model. The app is built with Streamlit for an interactive interface.

Project Structure
glaucoma-risk-prediction/
  1.glaucoma.csv              # Dataset (sample patient data)
  2.glaucoma_clf_model.pkl    # Trained ML model (must be added)
  3.app.py                    # Streamlit app for prediction
  4. requirements.txt          # Project dependencies
  5. README.md                 # Documentation

App Features:
User input via sliders and dropdowns for:
Age
Intraocular Pressure (IOP)
Cup-to-Disc Ratio (CDR)
Pachymetry (Corneal Thickness)
Visual Field Sensitivity & Specificity
Gender, Family History, Cataract Status, Angle Closure Status

Predictions:
No Glaucoma → ✅ Low risk, monitoring advised.
Glaucoma → ⚠️ High risk, immediate consultation recommended.

Dataset:
The dataset (glaucoma.csv) contains clinical features related to glaucoma risk, including intraocular pressure, age, pachymetry, and family history.

Future Improvements:
Enhance dataset with more patient records.
Add model training scripts.
Deploy online (Heroku/Streamlit Cloud).
Extend to image-based detection using retinal scans.


