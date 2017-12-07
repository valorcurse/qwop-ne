#!/bin/sh

unset GTK_IM_MODULE
unset QT4_IM_MODULE
unset CLUTTER_IM_MODULE
unset XMODIFIERS
unset DBUS_SESSION_BUS_ADDRESS

display=$1
shift

export DISPLAY=$display
exec "$@"