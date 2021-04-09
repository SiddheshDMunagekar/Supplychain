mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"munagekar.14121995sid@gmail.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml