#!/bin/bash
echo "============================================"
echo "  Nord Pool BESS Trading Platform"
echo "============================================"
echo ""
echo "Installing dependencies..."
pip install streamlit pandas numpy plotly
echo ""
echo "Starting app..."
streamlit run streamlit_app.py
