ARG BUILD_IMAGE
FROM node:24-bookworm-slim AS pi
RUN npm install --ignore-scripts --prefix /opt/pi @earendil-works/pi-coding-agent@0.85.1

FROM ${BUILD_IMAGE}
COPY --from=pi /usr/local/bin/node /usr/local/bin/node
COPY --from=pi /opt/pi /opt/pi
ENV PATH="/opt/pi/node_modules/.bin:${PATH}"
