FROM nixos/nix AS builder

WORKDIR /app

RUN mkdir -p /output/store

# Install batsim
RUN nix-env -f https://github.com/oar-team/nur-kapack/archive/master.tar.gz --profile /output/profile -iA batsim

# Copy all the run time dependencies into /output/store
RUN cp -va $(nix-store -qR /output/profile) /output/store

ENTRYPOINT [ "/bin/sh" ]

FROM python:3.10-slim AS builder2

WORKDIR /app

RUN apt update -y && apt install -y git && python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install git+https://github.com/tanaxer01/batsim-py \
                git+https://github.com/tanaxer01/GridGym   \
                torch gymnasium stable-baselines3

FROM python:3.10-slim

COPY --from=builder /output/store /nix/store
COPY --from=builder /output/profile /usr/.local

COPY --from=builder2 /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:/usr/.local/bin:$PATH"

CMD ["bash"]
