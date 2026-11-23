#!/usr/bin/env bash
# Phone connects directly to the car; no Internet, NAT or Ethernet changes.
# Run on the board: sudo bash scripts/setup_phone_ap.sh
# Override if needed: sudo env SSID=MyCar PASS=MyPassword123 bash scripts/setup_phone_ap.sh
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LC_ALL=C

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
[[ $(id -u) == 0 ]] || die 'Run with sudo on the board.'
for tool in nmcli rfkill dnsmasq systemctl loginctl; do
  command -v "$tool" >/dev/null || die "Missing command: $tool"
done

IFACE=wlan0
AP_IP=192.168.4.1
CON_NAME=line-follow-ap
# Edit these defaults, or pass SSID and PASS with sudo env as shown above.
SSID="${SSID:-QuectelPiM2}"
AP_PASS="${PASS:-pi12345678}"
UNIT=line-follow-ap-dhcp.service
CONF=/etc/dnsmasq.d/line-follow-ap.conf
[[ -n $SSID && ${#SSID} -le 32 ]] || die 'SSID must contain 1 to 32 bytes.'
[[ $AP_PASS =~ ^[[:print:]]{8,63}$ ]] || die 'PASS must contain 8 to 63 printable ASCII characters.'

# The old motion-control AP DHCP service must not race this one.
for other in car-ap-dhcp.service dnsmasq.service; do
  if systemctl is-active --quiet "$other"; then
    die "$other is active; resolve the existing DHCP service before continuing."
  fi
done
nmcli -g GENERAL.TYPE device show "$IFACE" | grep -qx wifi || die 'wlan0 is not a Wi-Fi device.'
nmcli -g WIFI-PROPERTIES.AP device show "$IFACE" | grep -qx yes || die 'Wi-Fi driver does not advertise AP support.'

if nmcli -g connection.id connection show "$CON_NAME" >/dev/null 2>&1; then
  [[ $(nmcli -g connection.interface-name connection show "$CON_NAME") == "$IFACE" ]] || die 'Existing line-follow-ap belongs to a different interface.'
  PROFILE_MODE=$(nmcli -g 802-11-wireless.mode connection show "$CON_NAME")
  if [[ $PROFILE_MODE != ap ]]; then
    # The first revision could create the profile, then fail before converting it
    # to an AP. Repair only that exact unused, unsecured infrastructure profile.
    [[ $PROFILE_MODE == infrastructure ]] || die 'Unexpected existing Wi-Fi mode.'
    [[ $(nmcli -g ipv4.method connection show "$CON_NAME") == auto ]] || die 'Existing profile has custom IP settings.'
    [[ -z $(nmcli -g ipv4.addresses connection show "$CON_NAME") ]] || die 'Existing profile has custom addresses.'
    [[ -z $(nmcli -g 802-11-wireless-security.key-mgmt connection show "$CON_NAME") ]] || die 'Existing profile has security settings; inspect it first.'
    [[ $(nmcli -g GENERAL.CONNECTION device show "$IFACE") != "$CON_NAME" ]] || die 'Existing profile is active; inspect it first.'
    printf 'Repairing the incomplete line-follow-ap profile from the earlier attempt.\n'
  fi
else
  nmcli connection add type wifi ifname "$IFACE" con-name "$CON_NAME" ssid "$SSID" autoconnect yes
fi

nmcli connection modify "$CON_NAME" \
  connection.interface-name "$IFACE" \
  connection.autoconnect yes connection.autoconnect-priority 999 \
  802-11-wireless.ssid "$SSID" \
  802-11-wireless.mode ap 802-11-wireless.band bg 802-11-wireless.channel 6 \
  802-11-wireless-security.key-mgmt wpa-psk \
  802-11-wireless-security.proto rsn \
  802-11-wireless-security.psk "$AP_PASS" \
  ipv4.method manual ipv4.addresses "$AP_IP/24" \
  ipv4.never-default yes ipv4.gateway '' ipv4.dns '' ipv6.method ignore

mkdir -p /etc/dnsmasq.d
cat > "$CONF" <<EOF
# DHCP only, restricted to the car's wireless AP. No Ethernet DHCP or NAT.
port=0
interface=$IFACE
bind-dynamic
except-interface=eth0
except-interface=lo
dhcp-range=192.168.4.10,192.168.4.100,255.255.255.0,12h
dhcp-option=3,$AP_IP
dhcp-option=6
dhcp-leasefile=/var/lib/misc/line-follow-ap.leases
log-dhcp
EOF
dnsmasq --test --conf-file="$CONF"
mkdir -p /var/lib/misc
cat > "/etc/systemd/system/$UNIT" <<EOF
[Unit]
Description=Line-follow car phone AP DHCP (wlan0 only)
After=NetworkManager.service
Wants=NetworkManager.service

[Service]
Type=simple
ExecStart=/usr/sbin/dnsmasq --keep-in-foreground --conf-file=$CONF
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
rfkill unblock wifi
nmcli radio wifi on
nmcli --wait 30 connection up "$CON_NAME"
systemctl enable "$UNIT"
systemctl restart "$UNIT"
systemctl is-active --quiet "$UNIT" || die 'AP DHCP did not start.'
[[ $(nmcli -g GENERAL.CONNECTION device show "$IFACE") == "$CON_NAME" ]] || die 'AP profile is not active.'
# The enabled user preview service must survive logout and boot without SSH.
APP_USER=${SUDO_USER:-}
[[ -n $APP_USER && $APP_USER != root ]] || die 'Run with sudo from the user who will own the preview service.'
id "$APP_USER" >/dev/null
loginctl enable-linger "$APP_USER"

printf '\nSSID: %s\nPassword: %s\nWeb: http://%s:8081/\n' "$SSID" "$AP_PASS" "$AP_IP"
printf 'AP auto-connect and DHCP auto-start are enabled. Re-running applies the configured SSID and PASS.\n'
printf 'Phone: choose to stay connected if warned that this Wi-Fi has no Internet.\n'
printf 'This script never starts line following. Use Start/Stop in the web page.\n'
