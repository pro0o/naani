# ollama3 llm server

make sure you have python and pip installed.

install the ollama package.
```
pip install ollama
```

install ollama3.
```
ollama run llama3
```

run the server.
```
python main.py
```

call the service.
```
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"content":"any query to ollama"}' \
  http://localhost:5000/recyclingData/tips

```