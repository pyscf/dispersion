#!/bin/bash

docker run --rm -v `pwd`:/io pyscf/pyscf-pypa-env:latest bash -c '
    pip wheel -v --no-deps --no-clean -w /root/wheelhouse /io &&
    auditwheel -v repair /root/wheelhouse/* -w /io/linux-wheels &&
    chown -R 1000:1000 /io/linux-wheels'
