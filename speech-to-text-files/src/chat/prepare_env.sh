#!/bin/bash

echo echo "load-module module-native-protocol-tcp auth-ip-acl=127.0.0.1 auth-anonymous=1" | tee -a /etc/pulse/default.pa
pulseaudio -k
pulseaudio --start
