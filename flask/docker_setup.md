# Docker Setup for Flask AdminLTE AJAX Application

## Recommended Docker Hub Image

For this Flask application with AdminLTE and AJAX functionality, I recommend using the official Python Docker image:

```
python:3.9-slim
```

This image is lightweight yet provides all the necessary components to run your Flask application.

## Dockerfile

Create a `Dockerfile` in the root of your project with the following content:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "src/main.py"]
```

## Requirements File

Ensure your `requirements.txt` file contains:

```
flask==3.1.1
```

## Docker Commands

### Build the Docker Image
```bash
docker build -t flask-adminlte-app .
```

### Run the Docker Container
```bash
docker run -p 5000:5000 flask-adminlte-app
```

The application will be accessible at http://localhost:5000

## Docker Compose (Optional)

For easier management, you can use Docker Compose. Create a `docker-compose.yml` file:

```yaml
version: '3'
services:
  web:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - .:/app
    environment:
      - FLASK_ENV=development
```

Then run:
```bash
docker-compose up
```

## Notes

- The application is configured to listen on all interfaces (0.0.0.0) which is required for Docker container access
- Port 5000 is exposed and mapped to the host
- For production deployment, consider using Gunicorn or uWSGI instead of the Flask development server
