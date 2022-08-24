from waitress import serve
import app
print("started server")
serve(app.app, host='0.0.0.0', port=5002)