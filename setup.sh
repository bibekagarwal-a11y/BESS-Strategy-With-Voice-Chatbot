mkdir -p ~/.streamlit

# Create config.toml for Streamlit
cat <<EOT > ~/.streamlit/config.toml
[server]
headless = true
port = $PORT
enableCORS = false
EOT
