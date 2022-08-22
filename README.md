# recsys-server
Recommendation system for restaurants and cafes

Launching the app:

 - build docker image for backend `docker build -f Dockerfile -t alitvintsev/recsys-backend .`
 - build docker image for frontend `docker build -f Dockerfile -t alitvintsev/recsys-frontend .`
 - run docker container for backend `docker run -d --name recsys-back -p 8000:8000 alitvintsev/recsys-backend`
 - run docker container for frontend `docker run -d --name recsys-front -p 8502:8502 alitvintsev/recsys-frontend`