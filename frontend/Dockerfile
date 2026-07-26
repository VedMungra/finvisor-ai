# Frontend: builds the Vite/React app, then serves the static output via nginx.
# Two stages keep the final image small (no Node.js runtime in the shipped image).

FROM node:20-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .

# Vite inlines import.meta.env.* at BUILD time, so the API base URL has to be known here --
# it cannot be injected later as a container environment variable, because by the time nginx
# is serving the bundle the value is already compiled into the JavaScript.
#
# It also cannot come from frontend/.env: .dockerignore excludes .env from the build context
# (correctly -- secrets shouldn't be baked into images). Without this ARG the build saw no
# VITE_API_URL at all and shipped a bundle that requested `undefined/chat`. docker-compose.yml
# supplies the value via build.args; the default below keeps a bare `docker build` working.
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL

RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
# SPA fallback: any unknown path serves index.html so client-side routing (if added later)
# doesn't 404 on refresh.
RUN echo 'server { listen 80; location / { root /usr/share/nginx/html; try_files $uri $uri/ /index.html; } }' \
    > /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
